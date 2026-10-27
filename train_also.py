"""CIFAR-100 / ResNet34: original ALSO or ALSO with adaptive group subsampling."""

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34

from also import ALSO
from sampler import NoReplacementSampler, WeightAdaptiveSampler

N_GROUPS = 100


def build_model():
    model = resnet34(weights=None, num_classes=N_GROUPS)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def stratified_split(targets):
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [], []
    for group in range(N_GROUPS):
        indices = torch.where(targets == group)[0]
        if len(indices) <= 20:
            raise ValueError(f"Group {group} needs more than 20 examples")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:20].tolist())
        train_indices.extend(indices[20:].tolist())
    return train_indices, val_indices


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def learning_rate_for_epoch(args, epoch):
    """ALSO uses optimizer.lr, so a torch scheduler alone would not change it."""
    drops = sum(epoch > milestone for milestone in args.milestones) if args.scheduler == "multistep" else 0
    return args.lr * args.lr_gamma ** drops


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    # Restore the successful 20261023_001356 run's training settings.
    # That run used AdamW; keeping these settings does not promise its accuracy with ALSO.
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["multistep", "none"], default="multistep")
    parser.add_argument("--milestones", type=int, nargs="*", default=[10, 15])
    parser.add_argument("--lr-gamma", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--sampling", choices=["adaptive", "without_replacement", "with_replacement"],
                        default="adaptive")
    parser.add_argument("--pi-lr", type=float, default=1e-3)
    parser.add_argument("--pi-decay", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["optimistic", "descent-ascent"], default="optimistic")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-group-fraction", type=float, default=0.1)
    parser.add_argument("--min-per-group", type=int, default=16)
    parser.add_argument("--weight-ema", type=float, default=0.5,
                        help="Weight smoothing: 0 disables it; larger values react more slowly")
    parser.add_argument("--compensation", type=float, default=0.25,
                        help="Model-weight bonus = compensation * discarded group fraction (0..1)")
    parser.add_argument("--pi-correction", choices=["frequency", "none"], default="frequency",
                        help="Correct the pi loss signal for changed group observation frequency")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Stop after two training batches")
    parser.add_argument("--no-save-model", action="store_true", help="Save metrics and sampling state only")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative")
    if args.crop_padding < 0 or args.warmup_epochs < 0 or args.min_per_group < 1:
        parser.error("Invalid crop-padding, warmup-epochs or min-per-group")
    if not 0 < args.min_group_fraction <= 1 or not 0 <= args.weight_ema < 1:
        parser.error("min-group-fraction must be in (0,1]; weight-ema in [0,1)")
    if not 0 <= args.compensation <= 1:
        parser.error("compensation must be in [0,1]")
    if (not 0 < args.lr_gamma <= 1 or any(m < 1 for m in args.milestones)
            or args.milestones != sorted(set(args.milestones))):
        parser.error("lr-gamma must be in (0,1]; milestones must be sorted, unique positive epochs")
    for name in ("lr", "pi_lr", "weight_decay", "pi_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for a small verification run.")
    device = torch.device(device_name)
    root = Path(__file__).resolve().parent
    data_root = args.data_root or root / "data"
    output_root = args.output_root or root / "runs"
    run_dir = output_root / datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_ALSO_{args.sampling}")
    run_dir.mkdir(parents=True, exist_ok=False)

    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=args.crop_padding),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_source = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=train_transform)
    val_source = datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=val_transform)
    targets = torch.tensor(train_source.targets)
    train_indices, val_indices = stratified_split(targets)
    torch.save({"train": train_indices, "validation": val_indices}, run_dir / "split.pt")
    train_dataset = Subset(train_source, train_indices)
    train_group_ids = targets[train_indices]  # Positions align with train_dataset.
    group_sizes = torch.bincount(train_group_ids, minlength=N_GROUPS)
    pi_uniform = torch.full((N_GROUPS,), 1 / N_GROUPS, device=device)
    if args.sampling == "adaptive":
        sampler = WeightAdaptiveSampler(
            train_group_ids, pi_uniform, seed=args.seed + 1,
            warmup_epochs=args.warmup_epochs, min_fraction=args.min_group_fraction,
            min_per_group=args.min_per_group, weight_ema=args.weight_ema,
            compensation=args.compensation,
        )
    elif args.sampling == "without_replacement":
        sampler = NoReplacementSampler(train_dataset, seed=args.seed + 1)
    else:
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset),
                                generator=torch.Generator().manual_seed(args.seed + 1))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed + 2), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_source, val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 3),
    )
    model = build_model().to(device)
    optimizer = ALSO(
        model.parameters(), n_groups=N_GROUPS, batch_size=args.batch_size,
        mode=args.mode, alpha=1.0, lr=args.lr, weight_decay=args.weight_decay,
        pi_lr=args.pi_lr, pi_decay=args.pi_decay, pi_init=pi_uniform, pi_reg=pi_uniform,
    )
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        optimizer="ALSO", n_groups=N_GROUPS, split_seed=42, alpha=1.0,
        architecture="torchvision_resnet34_cifar", pretrained=False,
        settings_reference_run="20261023_001356", settings_reference_commit="d961aaa",
        sampler_class=type(sampler).__name__, data_root=str(data_root), device=str(device),
        original_train_examples=len(train_dataset), validation_examples=len(val_indices),
        reference_weights=pi_uniform.cpu().tolist(), model_bonus_rule="1 + compensation * (1 - retention)",
        pi_correction_rule="overall_retention / group_retention; pi signal only",
        source_commit="5cea735", sampler_version=1,
        upstream_commit="bc4a5d8de99de382fc87632da46357280b16e6b9",
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save(optimizer.pi.detach().cpu(), run_dir / "pi_initial.pt")
    print("Модель: torchvision ResNet34, вход адаптирован под CIFAR")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}; устройство: {device}")
    print(f"Семплер: {type(sampler).__name__}; режим ALSO: {args.mode}")
    print(f"Train: {len(train_dataset)}, validation: {len(val_indices)}; результаты: {run_dir}")

    def save_model(path, epoch, accuracy):
        if not args.no_save_model:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_accuracy": accuracy, "pi": optimizer.pi.detach().cpu(),
                        "config": config, "checkpoint_purpose": "evaluation_only_not_training_resume"}, path)

    metric_fields = [
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds",
        "train_examples", "full_train_examples", "kept_fraction", "train_batches",
        "cumulative_train_examples", "train_seconds", "validation_seconds", "cumulative_seconds",
        "val_worst_group_accuracy", "pi_min", "pi_max", "pi_entropy", "max_model_bonus", "learning_rate",
    ]
    group_fields = [
        "epoch", "group", "group_size", "active_count", "retention", "pi_start", "pi_smoothed",
        "pi_end", "model_multiplier", "effective_weight_start", "pi_loss_multiplier",
        "train_seen", "train_mean_loss", "val_count", "val_accuracy",
    ]
    best_val_accuracy, cumulative_seconds, cumulative_examples = -1.0, 0.0, 0
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file, \
            (run_dir / "group_metrics.csv").open("w", newline="", encoding="utf-8") as group_file:
        writer, group_writer = csv.DictWriter(file, metric_fields), csv.DictWriter(group_file, group_fields)
        writer.writeheader()
        group_writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            epoch_start = time.perf_counter()
            optimizer.lr = learning_rate_for_epoch(args, epoch)
            pi_start = optimizer.pi.detach().cpu().clone()
            if isinstance(sampler, WeightAdaptiveSampler):
                sampler.set_epoch(epoch, pi_start)
                active_counts = sampler.active_counts.clone()
                retention = sampler.retention
                smoothed = sampler.smoothed_weights
                model_factors = sampler.model_weight_multipliers.to(device=device, dtype=optimizer.pi.dtype)
                pi_factors = sampler.pi_loss_multipliers.to(device=device, dtype=optimizer.pi.dtype)
                if args.pi_correction == "none":
                    pi_factors = torch.ones_like(pi_factors)
                # Both local and original dataset IDs make the selection auditable.
                torch.save({
                    "epoch": epoch, "active_subset_indices": sampler.active_indices,
                    "active_source_indices": torch.tensor(train_indices)[sampler.active_indices],
                    "group_sizes": group_sizes, "active_counts": active_counts,
                    "pi_start": pi_start, "smoothed_weights": smoothed,
                    "model_weight_multipliers": model_factors.cpu(),
                    "pi_loss_multipliers": pi_factors.cpu(),
                }, run_dir / f"sampler_epoch_{epoch:03d}.pt")
            else:
                active_counts, retention, smoothed = group_sizes, torch.ones(N_GROUPS), pi_start
                model_factors = torch.ones_like(optimizer.pi)
                pi_factors = torch.ones_like(optimizer.pi)
            print(f"Эпоха {epoch}: {len(sampler)}/{len(train_dataset)} изображений, "
                  f"{len(train_loader)} батчей, lr={optimizer.lr:.6g}")
            synchronize(device)
            train_start = time.perf_counter()
            model.train()
            train_loss_sum, train_correct, train_count = 0.0, 0, 0
            train_group_loss = torch.zeros(N_GROUPS, device=device)
            train_group_count = torch.zeros(N_GROUPS, dtype=torch.long, device=device)
            for step, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.loss_scale = N_GROUPS / labels.numel()  # Includes the short final batch.
                batch_log = {}

                def closure(weights, scale):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    raw_losses = nn.functional.cross_entropy(logits, labels, reduction="none")
                    scaled_losses = scale * raw_losses
                    weighted_loss = (weights * scaled_losses).sum()
                    if epoch == 1 and step == 1:
                        torch.testing.assert_close(weighted_loss, raw_losses.mean())
                    if not torch.isfinite(weighted_loss):
                        raise RuntimeError("Non-finite weighted loss")
                    weighted_loss.backward()
                    if args.smoke_test:
                        for parameter in model.parameters():
                            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                                raise RuntimeError("Non-finite model gradient")
                    batch_log["logits"] = logits.detach()
                    batch_log["losses"] = raw_losses.detach()
                    # Do not pass model-weighted losses into the pi update.
                    return scaled_losses.detach(), raw_losses.detach().mean().item()

                closure.device = device
                optimizer.step(closure, labels, model_weight_multipliers=model_factors,
                               pi_loss_multipliers=pi_factors)
                pi = optimizer.pi.detach()
                if not torch.isfinite(pi).all() or not (pi > 0).all():
                    raise RuntimeError("Invalid group weights")
                if args.smoke_test:
                    torch.testing.assert_close(pi.sum(), pi.new_tensor(1.0))
                    if not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("Non-finite model parameters")
                    if step == 2:
                        print("SMOKE TEST PASSED (2 batches; not a quality benchmark)")
                        return run_dir
                losses, logits = batch_log["losses"], batch_log["logits"]
                train_loss_sum += losses.sum().item()
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_count += labels.numel()
                train_group_loss.index_add_(0, labels, losses)
                train_group_count += torch.bincount(labels, minlength=N_GROUPS)
                if step % 100 == 0:
                    print(f"  Батч {step}/{len(train_loader)}")
            if train_count != len(sampler):
                raise RuntimeError("Processed sample count does not match sampler length")
            if isinstance(sampler, WeightAdaptiveSampler):
                torch.testing.assert_close(train_group_count.cpu(), active_counts)
            synchronize(device)
            train_seconds = time.perf_counter() - train_start
            validation_start = time.perf_counter()
            model.eval()
            val_loss_sum, val_correct, val_count = 0.0, 0, 0
            val_group_count = torch.zeros(N_GROUPS, dtype=torch.long, device=device)
            val_group_correct = torch.zeros_like(val_group_count)
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logits = model(images)
                    correct = logits.argmax(dim=1) == labels
                    val_loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                    val_correct += correct.sum().item()
                    val_count += labels.numel()
                    val_group_count += torch.bincount(labels, minlength=N_GROUPS)
                    val_group_correct.index_add_(0, labels, correct.long())
            synchronize(device)
            validation_seconds = time.perf_counter() - validation_start
            val_accuracy = val_correct / val_count
            val_group_accuracy = val_group_correct.float() / val_group_count
            pi_end = optimizer.pi.detach().cpu()
            torch.save(pi_end, run_dir / f"pi_epoch_{epoch:03d}.pt")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_model(run_dir / "model_best.pt", epoch, val_accuracy)
            if epoch == args.epochs:
                save_model(run_dir / "model_last.pt", epoch, val_accuracy)
            seen, group_loss = train_group_count.cpu(), train_group_loss.cpu()
            bonus, correction = model_factors.cpu(), pi_factors.cpu()
            for group in range(N_GROUPS):
                group_writer.writerow(dict(
                    epoch=epoch, group=group, group_size=int(group_sizes[group]),
                    active_count=int(active_counts[group]) if args.sampling != "with_replacement" else "",
                    retention=float(retention[group]) if args.sampling != "with_replacement" else "",
                    pi_start=float(pi_start[group]), pi_smoothed=float(smoothed[group]),
                    pi_end=float(pi_end[group]), model_multiplier=float(bonus[group]),
                    effective_weight_start=float(pi_start[group] * bonus[group]),
                    pi_loss_multiplier=float(correction[group]), train_seen=int(seen[group]),
                    train_mean_loss=float(group_loss[group] / seen[group]) if seen[group] else "",
                    val_count=int(val_group_count[group]), val_accuracy=float(val_group_accuracy[group]),
                ))
            group_file.flush()
            seconds = time.perf_counter() - epoch_start
            cumulative_seconds += seconds
            cumulative_examples += train_count
            writer.writerow(dict(
                epoch=epoch, train_loss=train_loss_sum / train_count, train_accuracy=train_correct / train_count,
                val_loss=val_loss_sum / val_count, val_accuracy=val_accuracy, seconds=seconds,
                train_examples=train_count, full_train_examples=len(train_dataset), kept_fraction=train_count / len(train_dataset),
                train_batches=len(train_loader), cumulative_train_examples=cumulative_examples,
                train_seconds=train_seconds, validation_seconds=validation_seconds, cumulative_seconds=cumulative_seconds,
                val_worst_group_accuracy=float(val_group_accuracy.min()), pi_min=float(pi_end.min()),
                pi_max=float(pi_end.max()), pi_entropy=float(-(pi_end * pi_end.log()).sum()),
                max_model_bonus=float(bonus.max() - 1), learning_rate=optimizer.lr,
            ))
            file.flush()
            print(f"Эпоха {epoch}/{args.epochs} | train acc {train_correct / train_count:.1%} | "
                  f"val acc {val_accuracy:.2%} | best {best_val_accuracy:.2%} | "
                  f"оставлено {train_count / len(train_dataset):.1%} | {seconds:.1f} сек.")
    return run_dir


if __name__ == "__main__":
    main()
