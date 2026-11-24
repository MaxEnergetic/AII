"""MedMNIST training adapter for AII ALSO and WeightAdaptiveSampler.

New adapter; bundled also.py and sampler.py are unchanged upstream snapshots.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import medmnist
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34
from sklearn.metrics import accuracy_score, f1_score

from also import ALSO
from sampler import NoReplacementSampler, WeightAdaptiveSampler


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_model(channels, outputs):
    model = resnet34(weights=None, num_classes=outputs)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def make_groups(labels, config):
    if config["dataset"] == "pathmnist":
        ids = torch.as_tensor(labels, dtype=torch.long).flatten()
        assert set(ids.tolist()) == set(range(9))
        return ids
    # Disjoint, fixed random IMAGE groups, not diagnosis groups.
    # All 14 labels remain in the loss; each image appears only once.
    n = len(labels)
    generator = torch.Generator().manual_seed(config["group_seed"])
    order = torch.randperm(n, generator=generator)
    ids = torch.empty(n, dtype=torch.long)
    ids[order] = torch.arange(n) // config["random_group_size"]
    return ids


class GroupedDataset(Dataset):
    def __init__(self, source, group_ids):
        self.source, self.group_ids = source, group_ids

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, target = self.source[index]
        return image, target, self.group_ids[index], index


def prepare_data(config):
    flag = config["dataset"]
    info = medmnist.INFO[flag]
    data_class = getattr(medmnist, info["python_class"])
    root = Path(config["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    channels = info["n_channels"]
    # Deterministic preprocessing only. No crop, flip, resize, noise or color transforms.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])
    sets = {
        split: data_class(split=split, root=str(root), transform=transform,
                          size=28, as_rgb=False, download=config.get("download", False))
        for split in ("train", "val", "test")
    }
    for split, dataset in sets.items():
        assert len(dataset) == info["n_samples"][split], (split, len(dataset))
        image, target = dataset[0]
        assert tuple(image.shape) == (channels, 28, 28)
        if flag == "chestmnist":
            labels = np.asarray(dataset.labels)
            assert labels.shape == (len(dataset), 14)
            assert np.isin(labels, [0, 1]).all()
        else:
            assert np.asarray(target).size == 1
    return sets


def per_image_loss(logits, targets, multilabel):
    if multilabel:
        # BCEWithLogitsLoss = numerically stable sigmoid + BCE.
        # Shape [B, 14] -> [B]; average over all labels, including negative ones.
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none").mean(dim=1)
    return nn.functional.cross_entropy(logits, targets.long().reshape(-1), reduction="none")


def prediction_metrics(targets, probabilities, multilabel, threshold=0.5):
    if multilabel:
        predictions = (probabilities >= threshold).astype(np.int64)
        return {
            "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(targets, predictions, average="micro", zero_division=0)),
            "predicted_positive_fraction": float(predictions.mean()),
        }
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets.reshape(-1), predictions)),
        "f1_macro": float(f1_score(targets.reshape(-1), predictions, labels=list(range(9)),
                                   average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, device, multilabel, threshold):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_targets, all_probabilities = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        losses = per_image_loss(logits, targets, multilabel)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Non-finite evaluation loss")
        total_loss += losses.sum().item()
        total_count += len(images)
        probabilities = logits.sigmoid() if multilabel else logits.softmax(dim=1)
        all_targets.append(targets.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    metrics = prediction_metrics(targets, probabilities, multilabel, threshold)
    metrics["loss"] = total_loss / total_count
    return metrics, targets, probabilities


def is_better(metrics, best_metrics, selection_metric):
    # Deterministic tie break: lower validation loss, then earlier epoch.
    if best_metrics is None:
        return True
    return (metrics[selection_metric], -metrics["loss"]) > (
        best_metrics[selection_metric], -best_metrics["loss"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(config, method, seed, run_dir, datasets=None, model_factory=build_model):
    assert method in ("also", "adaptive")
    assert config["dataset"] in ("pathmnist", "chestmnist")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = prepare_data(config) if datasets is None else datasets
    multilabel = config["dataset"] == "chestmnist"
    channels, outputs = (1, 14) if multilabel else (3, 9)
    selection_metric = "f1_macro" if multilabel else "accuracy"
    groups = make_groups(sets["train"].labels, config)
    group_sizes = torch.bincount(groups)
    n_groups = len(group_sizes)
    grouped = GroupedDataset(sets["train"], groups)
    reference = torch.full((n_groups,), 1.0 / n_groups, device=device)
    if method == "adaptive":
        sampler = WeightAdaptiveSampler(
            groups, reference, seed=seed + 1, warmup_epochs=config["warmup_epochs"],
            min_fraction=config["min_group_fraction"], min_per_group=config["min_per_group"],
            compensation=config["compensation"])
    else:
        sampler = NoReplacementSampler(grouped, seed=seed + 1)
    loader_kwargs = dict(batch_size=config["batch_size"], num_workers=config["num_workers"],
                         pin_memory=device.type == "cuda", drop_last=False)
    train_loader = DataLoader(grouped, sampler=sampler, **loader_kwargs,
                              generator=torch.Generator().manual_seed(seed + 2))
    eval_loaders = {
        split: DataLoader(sets[split], shuffle=False, **loader_kwargs,
                          generator=torch.Generator().manual_seed(seed + 3 + i))
        for i, split in enumerate(("train", "val", "test"))
    }
    model = model_factory(channels, outputs).to(device)
    optimizer = ALSO(model.parameters(), n_groups=n_groups, batch_size=config["batch_size"],
                     mode="optimistic", alpha=1.0, lr=config["lr"],
                     weight_decay=config["weight_decay"], pi_lr=config["pi_lr"],
                     pi_decay=config["pi_decay"], pi_init=reference, pi_reg=reference)
    initial_hash = hashlib.sha256()
    for tensor in model.state_dict().values():
        initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    run_config = dict(config, method=method, seed=seed, optimizer="ALSO", device=str(device),
                      selection_metric="val_" + selection_metric, n_groups=n_groups,
                      initial_model_sha256=initial_hash.hexdigest(),
                      checkpoint_purpose="evaluation_only_not_training_resume",
                      versions={"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                                "medmnist": medmnist.__version__},
                      train_size=len(grouped), val_size=len(sets["val"]), test_size=len(sets["test"]))
    write_json(run_dir / "config.json", run_config)
    torch.save({"group_ids": groups, "group_sizes": group_sizes,
                "train_labels": torch.as_tensor(sets["train"].labels)}, run_dir / "groups.pt")
    best_metrics, best_epoch = None, None
    history = []
    train_seconds_sum = 0.0
    training_started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        pi_start = optimizer.pi.detach().cpu().clone()
        if method == "adaptive":
            sampler.set_epoch(epoch, pi_start)
            active_counts = sampler.active_counts.clone()
            model_factors = sampler.model_weight_multipliers.to(device, dtype=reference.dtype)
            pi_factors = sampler.pi_loss_multipliers.to(device, dtype=reference.dtype)
            active_indices = sampler.active_indices.clone()
        else:
            active_counts = group_sizes.clone()
            model_factors = torch.ones_like(reference)
            pi_factors = torch.ones_like(reference)
            active_indices = torch.arange(len(grouped))
        model.train()
        loss_sum, objective_sum, seen = 0.0, 0.0, 0
        seen_groups = torch.zeros(n_groups, dtype=torch.long)
        synchronize(device)
        train_started = time.perf_counter()
        for step, (images, targets, batch_groups, indices) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            batch_groups = batch_groups.to(device)
            batch_size = len(images)  # NOT targets.numel(): multilabel targets are [B, 14].
            optimizer.loss_scale = n_groups / batch_size
            batch_log = {}

            def closure(weights, scale):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                raw_losses = per_image_loss(logits, targets, multilabel)
                scaled_losses = scale * raw_losses
                objective = (weights * scaled_losses).sum()
                if epoch == 1 and step == 1:
                    torch.testing.assert_close(objective, raw_losses.mean())
                if not torch.isfinite(objective):
                    raise RuntimeError("Non-finite weighted training loss")
                objective.backward()
                batch_log["raw_sum"] = raw_losses.detach().sum().item()
                batch_log["objective"] = objective.detach().item()
                # pi update gets scaled UNWEIGHTED per-image losses, as in AII.
                return scaled_losses.detach(), raw_losses.detach().mean().item()

            closure.device = device
            optimizer.step(closure, batch_groups, model_weight_multipliers=model_factors,
                           pi_loss_multipliers=pi_factors)
            if not torch.isfinite(optimizer.pi).all() or not (optimizer.pi > 0).all():
                raise RuntimeError("Invalid ALSO weights")
            loss_sum += batch_log["raw_sum"]
            objective_sum += batch_log["objective"] * batch_size
            seen += batch_size
            seen_groups += torch.bincount(batch_groups.cpu(), minlength=n_groups)
            if step % config["print_every"] == 0:
                print(f"epoch {epoch}: batch {step}/{len(train_loader)}", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - train_started
        train_seconds_sum += train_seconds
        assert seen == len(sampler)
        torch.testing.assert_close(seen_groups, active_counts)
        val_metrics, _, _ = evaluate(model, eval_loaders["val"], device, multilabel, config["threshold"])
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_objective=objective_sum / seen,
                   train_examples=seen, kept_fraction=seen / len(grouped), train_seconds=train_seconds,
                   **{"val_" + key: value for key, value in val_metrics.items()})
        if config["full_train_eval"]:
            eval_start = time.perf_counter()
            train_metrics, _, _ = evaluate(model, eval_loaders["train"], device, multilabel, config["threshold"])
            row.update({"train_eval_" + key: value for key, value in train_metrics.items()})
            row["full_train_eval_seconds"] = time.perf_counter() - eval_start
        if is_better(val_metrics, best_metrics, selection_metric):
            best_metrics, best_epoch = dict(val_metrics), epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_metrics": best_metrics, "threshold": config["threshold"],
                        "pi": optimizer.pi.detach().cpu(), "config": run_config}, run_dir / "model_best.pt")
        torch.save({"epoch": epoch, "active_indices": active_indices,
                    "active_counts": active_counts, "pi_start": pi_start,
                    "pi_end": optimizer.pi.detach().cpu(),
                    "model_weight_multipliers": model_factors.cpu(),
                    "pi_loss_multipliers": pi_factors.cpu()}, run_dir / f"sampler_epoch_{epoch:03d}.pt")
        synchronize(device)
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        history.append(row)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch {epoch}/{config['epochs']} | train loss {row['train_loss']:.5f} | "
              f"val loss {val_metrics['loss']:.5f} | val {selection_metric} "
              f"{val_metrics[selection_metric]:.4f} | best epoch {best_epoch} | "
              f"kept {row['kept_fraction']:.1%} | {row['epoch_seconds']:.1f}s", flush=True)
    training_wall_seconds = time.perf_counter() - training_started
    # Test is evaluated exactly once, after reloading the validation-selected checkpoint.
    checkpoint = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=True)
    assert checkpoint["epoch"] == best_epoch
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, targets, probabilities = evaluate(
        model, eval_loaders["test"], device, multilabel, checkpoint["threshold"])
    np.savez_compressed(run_dir / "test_predictions.npz", targets=targets, probabilities=probabilities,
                        threshold=checkpoint["threshold"], best_epoch=best_epoch)
    if multilabel:
        per_label = f1_score(targets, probabilities >= checkpoint["threshold"], average=None, zero_division=0)
        labels = medmnist.INFO[config["dataset"]]["label"]
        with (run_dir / "test_f1_per_label.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["label_id", "label", "f1", "positive_support"])
            for i, score in enumerate(per_label):
                writer.writerow([i, labels[str(i)], float(score), int(targets[:, i].sum())])
    result = dict(method=method, seed=seed, best_epoch=best_epoch,
                  best_val_score=best_metrics[selection_metric], selection_metric=selection_metric,
                  **{"test_" + key: value for key, value in test_metrics.items()},
                  training_seconds=train_seconds_sum, training_wall_seconds=training_wall_seconds,
                  total_train_examples=sum(r["train_examples"] for r in history),
                  mean_kept_fraction=float(np.mean([r["kept_fraction"] for r in history])))
    write_json(run_dir / "result.json", result)
    print("TEST (best validation epoch): " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["also", "adaptive"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.method, args.seed, args.run_dir)
