"""CIFAR-100 / ResNet34 with fixed, class-independent random ALSO groups.

Shuffle all training positions once with --group-seed, then assign one group ID
per --batch-size items. The final group may be smaller. Group IDs stay fixed
throughout training; they are independent of the true CIFAR classification labels.
ALSO and the sampler use group IDs; cross-entropy and accuracy use true labels.
The sampler still shuffles individual images, so a training minibatch can contain
several groups. This script does not require changes to also.py or sampler.py.
"""

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
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34

from also import ALSO
from sampler import NoReplacementSampler, WeightAdaptiveSampler

N_CLASSES = 100
VAL_PER_CLASS = 20  # Preserve the 48,000 train / 2,000 validation split.


def build_model():
    model = resnet34(weights=None, num_classes=N_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def stratified_split(targets):
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [], []
    for group in range(N_CLASSES):
        indices = torch.where(targets == group)[0]
        if len(indices) <= VAL_PER_CLASS:
            raise ValueError(f"Class {group} needs more than {VAL_PER_CLASS} examples")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:VAL_PER_CLASS].tolist())
        train_indices.extend(indices[VAL_PER_CLASS:].tolist())
    return train_indices, val_indices


def make_group_ids(n_items, batch_size, seed=123):
    """Assign fixed random groups of batch_size without reading class labels.

    Returned positions align with the training Subset, not CIFAR source IDs.
    A private generator leaves the model RNG untouched. Each item belongs to
    exactly one group; only the last group can contain fewer than batch_size.
    """
    if n_items < 1 or int(n_items) != n_items:
        raise ValueError("n_items must be a positive integer")
    if batch_size < 1 or int(batch_size) != batch_size:
        raise ValueError("batch_size must be a positive integer")
    n_items, batch_size = int(n_items), int(batch_size)
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(n_items, generator=generator)
    group_ids = torch.empty(n_items, dtype=torch.long)
    group_ids[positions] = torch.arange(n_items) // batch_size
    return group_ids


class GroupedDataset(Dataset):
    """Keep classification labels separate from fixed ALSO group IDs."""

    def __init__(self, dataset, group_ids):
        self.dataset = dataset
        self.group_ids = torch.as_tensor(group_ids, dtype=torch.long, device="cpu").clone()
        if self.group_ids.ndim != 1 or len(self.group_ids) != len(dataset):
            raise ValueError("One group ID is required per training item")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.group_ids[index], index


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def learning_rate_for_epoch(args, epoch):
    """ALSO uses optimizer.lr, so a torch scheduler alone would not change it."""
    drops = sum(epoch > milestone for milestone in args.milestones) if args.scheduler == "multistep" else 0
    return args.lr * args.lr_gamma ** drops


def save_distribution_plots(run_dir, per_epoch=False):
    """Write 2 PNGs: unique images in the final epoch, by class and ALSO group.

    Can also be called on an existing run with groups.pt and the metrics CSVs.
    Each dataset index counts once; augmentations and repeated draws do not add images.
    Rendering runs after training and is excluded from recorded epoch times.
    Set per_epoch=True to write 2 PNGs for every completed epoch.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    run_dir = Path(run_dir)
    assignment = torch.load(run_dir / "groups.pt", map_location="cpu", weights_only=True)
    n_classes = int(assignment["n_classes"])
    sizes = {
        "classes": torch.bincount(assignment["train_labels"], minlength=n_classes).numpy(),
        "groups": assignment["group_sizes"].numpy(),
    }
    histories = {}
    for kind, filename, id_field in (
        ("classes", "class_metrics.csv", "class_id"),
        ("groups", "group_metrics.csv", "group"),
    ):
        history = {}
        with (run_dir / filename).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not {"train_unique"}.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{filename} has no unique-image counts. Draw counts cannot be used "
                    "instead; use results from the updated training script."
                )
            for row in reader:
                counts = history.setdefault(int(row["epoch"]), np.full(len(sizes[kind]), -1, dtype=np.int64))
                index = int(row[id_field])
                if counts[index] != -1:
                    raise ValueError(f"Duplicate {kind} record in {filename}")
                counts[index] = int(row["train_unique"])
        histories[kind] = history
    # Only render epochs fully recorded in both files (also useful after interruption).
    epochs = sorted(set(histories["classes"]) & set(histories["groups"]))
    epochs = [epoch for epoch in epochs
              if all((histories[kind][epoch] >= 0).all() for kind in sizes)]
    if not epochs:
        return []
    for epoch in epochs:
        if histories["classes"][epoch].sum() != histories["groups"][epoch].sum():
            raise ValueError(f"Class/group counts disagree in epoch {epoch}")
        for kind, baseline in sizes.items():
            counts = histories[kind][epoch]
            if ((counts < 0) | (counts > baseline)).any():
                raise ValueError(f"Invalid unique-image counts for {kind}, epoch {epoch}")
    plot_dir = run_dir / "histograms"
    plot_dir.mkdir(exist_ok=True)
    # Remove obsolete generated charts when replotting an earlier run.
    for filename in ("classes_total.png", "groups_total.png"):
        (plot_dir / filename).unlink(missing_ok=True)
    paths = []
    for kind, baseline in sizes.items():
        # Common epoch scale makes reductions and changes comparable across PNGs.
        epoch_max = max(int(baseline.max()), *(int(histories[kind][e].max()) for e in epochs))
        chart_epochs = epochs if per_epoch else epochs[-1:]
        charts = [(f"epoch_{epoch:03d}", f"Epoch {epoch}", histories[kind][epoch], baseline, epoch_max)
                  for epoch in chart_epochs]
        for suffix, title, counts, reference, maximum in charts:
            figure = Figure(figsize=(16 if len(baseline) > 30 else 11, 5), layout="constrained")
            FigureCanvasAgg(figure)
            ax = figure.subplots()
            x = np.arange(len(baseline))
            ax.bar(x - 0.2, reference, width=0.4, color="#1f77b4", label="Original data (before sampling)")
            ax.bar(x + 0.2, counts, width=0.4, color="#ff7f0e", label="Selected data (unique images)")
            label = "CIFAR class ID" if kind == "classes" else "ALSO group ID"
            ax.set(title=f"{title}: {label} distribution", xlabel=label,
                   ylabel="Number of unique images", ylim=(0, max(1, maximum) * 1.15),
                   xlim=(-0.75, len(baseline) - 0.25))
            ticks = np.unique(np.r_[np.arange(0, len(baseline), max(1, math.ceil(len(baseline) / 40))), len(baseline) - 1])
            ax.set_xticks(ticks)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper right", fontsize=8)
            path = plot_dir / f"{kind}_{suffix}.png"
            figure.savefig(path, dpi=150)
            figure.clear()
            paths.append(path)
    return paths


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
    parser.add_argument("--group-seed", type=int, default=123,
                        help="Independent seed for fixed random groups across all training classes")
    parser.add_argument("--sampling", choices=["adaptive", "without_replacement", "with_replacement"],
                        default="adaptive")
    parser.add_argument("--pi-lr", type=float, default=1e-3)
    parser.add_argument("--pi-decay", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["optimistic", "descent-ascent"], default="optimistic")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-group-fraction", type=float, default=0.1)
    parser.add_argument("--min-per-group", type=int, default=16)
    parser.add_argument("--compensation", type=float, default=0.25,
                        help="Model-weight bonus = compensation * discarded group fraction (0..1)")
    parser.add_argument("--pi-correction", choices=["frequency", "none"], default="frequency",
                        help="Correct the pi loss signal for changed group observation frequency")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Stop after two training batches")
    parser.add_argument("--no-save-model", action="store_true", help="Save metrics and sampling state only")
    parser.add_argument("--histograms", type=int, choices=[0, 1], default=1,
                        help="Save two distribution PNGs after training: 1=yes, 0=no (default: 1)")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative")
    if args.warmup_epochs < 0 or args.min_per_group < 1:
        parser.error("Invalid warmup-epochs or min-per-group")
    if not 0 < args.min_group_fraction <= 1:
        parser.error("min-group-fraction must be in (0,1]")
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
    if args.histograms:
        # Fail before training if requested plots cannot be rendered.
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
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
    run_dir = output_root / datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_ALSO_batched_{args.sampling}")
    run_dir.mkdir(parents=True, exist_ok=False)

    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_source = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=train_transform)
    val_source = datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=val_transform)
    targets = torch.tensor(train_source.targets)
    train_indices, val_indices = stratified_split(targets)
    torch.save({"train": train_indices, "validation": val_indices}, run_dir / "split.pt")
    train_labels = targets[train_indices]
    train_group_ids = make_group_ids(len(train_indices), args.batch_size, args.group_seed)
    n_groups = (len(train_indices) + args.batch_size - 1) // args.batch_size
    train_dataset = GroupedDataset(Subset(train_source, train_indices), train_group_ids)
    group_sizes = torch.bincount(train_group_ids, minlength=n_groups)
    # Save the exact mapping once; every epoch uses these same random groups.
    torch.save({
        "train_source_indices": torch.tensor(train_indices),
        "train_labels": train_labels, "train_group_ids": train_group_ids,
        "group_sizes": group_sizes, "n_classes": N_CLASSES, "n_groups": n_groups,
        "group_size": args.batch_size, "group_seed": args.group_seed,
        "group_rule": "group_ids[randperm(n_train)] = arange(n_train) // batch_size",
    }, run_dir / "groups.pt")
    pi_uniform = torch.full((n_groups,), 1 / n_groups, device=device)
    if args.sampling == "adaptive":
        sampler = WeightAdaptiveSampler(
            train_group_ids, pi_uniform, seed=args.seed + 1,
            warmup_epochs=args.warmup_epochs, min_fraction=args.min_group_fraction,
            min_per_group=args.min_per_group,
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
        model.parameters(), n_groups=n_groups, batch_size=args.batch_size,
        mode=args.mode, alpha=1.0, lr=args.lr, weight_decay=args.weight_decay,
        pi_lr=args.pi_lr, pi_decay=args.pi_decay, pi_init=pi_uniform, pi_reg=pi_uniform,
    )
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        optimizer="ALSO", dataset="CIFAR100", n_classes=N_CLASSES, n_groups=n_groups,
        split_seed=42, val_per_class=VAL_PER_CLASS, alpha=1.0,
        grouping="fixed_random_across_classes", group_size=args.batch_size, group_assignment_file="groups.pt",
        validation_grouping="original_classes",
        train_augmentation=True,
        legacy_val_worst_group_accuracy="alias of val_worst_class_accuracy",
        architecture="torchvision_resnet34_cifar", pretrained=False,
        settings_reference_run="20261023_001356", settings_reference_commit="d961aaa",
        sampler_class=type(sampler).__name__, data_root=str(data_root), device=str(device),
        original_train_examples=len(train_dataset), validation_examples=len(val_indices),
        reference_weights=pi_uniform.cpu().tolist(), model_bonus_rule="1 + compensation * (1 - retention)",
        pi_correction_rule="overall_retention / group_retention; pi signal only",
        training_script_version="cifar100_fixed_random_batch_sized_groups_v1", sampler_version=3,
        adaptive_count_rule="previous active count * current weight / previous epoch weight",
        upstream_commit="bc4a5d8de99de382fc87632da46357280b16e6b9",
        histogram_directory="histograms" if args.histograms else None,
        histogram_count_rule="2: final epoch, by class and group" if args.histograms else "0 (disabled)",
        histogram_counts="unique dataset indices within each epoch",
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save(optimizer.pi.detach().cpu(), run_dir / "pi_initial.pt")
    print("Датасет: CIFAR-100; модель: torchvision ResNet34, вход адаптирован под CIFAR")
    print(f"Классов модели: {N_CLASSES}; случайных групп ALSO: {n_groups}; "
          f"размер группы={args.batch_size}, group seed={args.group_seed}")
    print(f"Размеры групп: {int(group_sizes.min())}..{int(group_sizes.max())}; "
          "группы фиксированы и не зависят от классов CIFAR")
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
        "val_worst_class_accuracy", "val_worst_group_accuracy", "pi_min", "pi_max", "pi_entropy",
        "max_model_bonus", "learning_rate",
    ]
    group_fields = [
        "epoch", "group", "group_size", "active_count", "retention",
        "pi_start",
        "pi_end", "model_multiplier", "effective_weight_start", "pi_loss_multiplier",
        "train_seen", "train_mean_loss", "train_unique",
    ]
    class_fields = [
        "epoch", "class_id", "train_seen", "train_mean_loss", "val_count", "val_accuracy",
        "train_unique",
    ]
    best_val_accuracy, cumulative_seconds, cumulative_examples = -1.0, 0.0, 0
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file, \
            (run_dir / "group_metrics.csv").open("w", newline="", encoding="utf-8") as group_file, \
            (run_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as class_file:
        writer, group_writer = csv.DictWriter(file, metric_fields), csv.DictWriter(group_file, group_fields)
        class_writer = csv.DictWriter(class_file, class_fields)
        writer.writeheader()
        group_writer.writeheader()
        class_writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            epoch_start = time.perf_counter()
            optimizer.lr = learning_rate_for_epoch(args, epoch)
            pi_start = optimizer.pi.detach().cpu().clone()
            if isinstance(sampler, WeightAdaptiveSampler):
                sampler.set_epoch(epoch, pi_start)
                active_counts = sampler.active_counts.clone()
                retention = sampler.retention
                model_factors = sampler.model_weight_multipliers.to(device=device, dtype=optimizer.pi.dtype)
                pi_factors = sampler.pi_loss_multipliers.to(device=device, dtype=optimizer.pi.dtype)
                if args.pi_correction == "none":
                    pi_factors = torch.ones_like(pi_factors)
                # Both local and original dataset IDs make the selection auditable.
                torch.save({
                    "epoch": epoch, "active_subset_indices": sampler.active_indices,
                    "active_source_indices": torch.tensor(train_indices)[sampler.active_indices],
                    "group_sizes": group_sizes, "active_counts": active_counts,
                    "pi_start": pi_start,
                    "model_weight_multipliers": model_factors.cpu(),
                    "pi_loss_multipliers": pi_factors.cpu(),
                }, run_dir / f"sampler_epoch_{epoch:03d}.pt")
            else:
                active_counts, retention = group_sizes, torch.ones(n_groups)
                model_factors = torch.ones_like(optimizer.pi)
                pi_factors = torch.ones_like(optimizer.pi)
            print(f"Эпоха {epoch}: {len(sampler)}/{len(train_dataset)} изображений, "
                  f"{len(train_loader)} батчей, lr={optimizer.lr:.6g}")
            synchronize(device)
            train_start = time.perf_counter()
            model.train()
            train_loss_sum, train_correct, train_count = 0.0, 0, 0
            train_group_loss = torch.zeros(n_groups, device=device)
            train_group_count = torch.zeros(n_groups, dtype=torch.long, device=device)
            # Random groups mix classes, so class statistics need true labels.
            train_class_loss = torch.zeros(N_CLASSES, device=device)
            train_class_count = torch.zeros(N_CLASSES, dtype=torch.long, device=device)
            epoch_seen_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
            for step, (images, labels, group_ids, sample_indices) in enumerate(train_loader, start=1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                group_ids = group_ids.to(device, non_blocking=True)
                optimizer.loss_scale = n_groups / labels.numel()  # Includes the short final batch.
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
                optimizer.step(closure, group_ids, model_weight_multipliers=model_factors,
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
                epoch_seen_mask[sample_indices] = True
                train_group_loss.index_add_(0, group_ids, losses)
                train_group_count += torch.bincount(group_ids, minlength=n_groups)
                train_class_loss.index_add_(0, labels, losses)
                train_class_count += torch.bincount(labels, minlength=N_CLASSES)
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
            val_class_count = torch.zeros(N_CLASSES, dtype=torch.long, device=device)
            val_class_correct = torch.zeros_like(val_class_count)
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logits = model(images)
                    correct = logits.argmax(dim=1) == labels
                    val_loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                    val_correct += correct.sum().item()
                    val_count += labels.numel()
                    val_class_count += torch.bincount(labels, minlength=N_CLASSES)
                    val_class_correct.index_add_(0, labels, correct.long())
            synchronize(device)
            validation_seconds = time.perf_counter() - validation_start
            val_accuracy = val_correct / val_count
            val_class_accuracy = val_class_correct.float() / val_class_count
            pi_end = optimizer.pi.detach().cpu()
            torch.save(pi_end, run_dir / f"pi_epoch_{epoch:03d}.pt")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_model(run_dir / "model_best.pt", epoch, val_accuracy)
            if epoch == args.epochs:
                save_model(run_dir / "model_last.pt", epoch, val_accuracy)
            unique_groups = torch.bincount(train_group_ids[epoch_seen_mask], minlength=n_groups)
            unique_classes = torch.bincount(train_labels[epoch_seen_mask], minlength=N_CLASSES)
            seen, group_loss = train_group_count.cpu(), train_group_loss.cpu()
            bonus, correction = model_factors.cpu(), pi_factors.cpu()
            for group in range(n_groups):
                group_writer.writerow(dict(
                    epoch=epoch, group=group, group_size=int(group_sizes[group]),
                    train_unique=int(unique_groups[group]),
                    active_count=int(active_counts[group]) if args.sampling != "with_replacement" else "",
                    retention=float(retention[group]) if args.sampling != "with_replacement" else "",
                    pi_start=float(pi_start[group]),
                    pi_end=float(pi_end[group]), model_multiplier=float(bonus[group]),
                    effective_weight_start=float(pi_start[group] * bonus[group]),
                    pi_loss_multiplier=float(correction[group]), train_seen=int(seen[group]),
                    train_mean_loss=float(group_loss[group] / seen[group]) if seen[group] else "",
                ))
            group_file.flush()
            class_seen = train_class_count.cpu()
            class_loss = train_class_loss.cpu()
            val_counts_cpu = val_class_count.cpu()
            val_accuracy_cpu = val_class_accuracy.cpu()
            for class_id in range(N_CLASSES):
                class_writer.writerow(dict(
                    epoch=epoch, class_id=class_id, train_seen=int(class_seen[class_id]),
                    train_unique=int(unique_classes[class_id]),
                    train_mean_loss=(float(class_loss[class_id] / class_seen[class_id])
                                     if class_seen[class_id] else ""),
                    val_count=int(val_counts_cpu[class_id]),
                    val_accuracy=float(val_accuracy_cpu[class_id]),
                ))
            class_file.flush()
            seconds = time.perf_counter() - epoch_start
            cumulative_seconds += seconds
            cumulative_examples += train_count
            writer.writerow(dict(
                epoch=epoch, train_loss=train_loss_sum / train_count, train_accuracy=train_correct / train_count,
                val_loss=val_loss_sum / val_count, val_accuracy=val_accuracy, seconds=seconds,
                train_examples=train_count, full_train_examples=len(train_dataset), kept_fraction=train_count / len(train_dataset),
                train_batches=len(train_loader), cumulative_train_examples=cumulative_examples,
                train_seconds=train_seconds, validation_seconds=validation_seconds, cumulative_seconds=cumulative_seconds,
                val_worst_class_accuracy=float(val_class_accuracy.min()),
                # Compatibility with old readers: this remains a CLASS-level metric.
                val_worst_group_accuracy=float(val_class_accuracy.min()), pi_min=float(pi_end.min()),
                pi_max=float(pi_end.max()), pi_entropy=float(-(pi_end * pi_end.log()).sum()),
                max_model_bonus=float(bonus.max() - 1), learning_rate=optimizer.lr,
            ))
            file.flush()
            print(f"Эпоха {epoch}/{args.epochs} | train acc {train_correct / train_count:.1%} | "
                  f"val acc {val_accuracy:.2%} | best {best_val_accuracy:.2%} | "
                  f"оставлено {train_count / len(train_dataset):.1%} | {seconds:.1f} сек.")
    if args.histograms:
        plots = save_distribution_plots(run_dir)
        print(f"Гистограммы: {len(plots)} PNG в {run_dir / 'histograms'}")
    return run_dir


if __name__ == "__main__":
    main()

