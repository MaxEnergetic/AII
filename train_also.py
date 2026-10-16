import csv
import json
from also import ALSO
import time
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader, Subset, RandomSampler

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import argparse

from model import ResNet32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    
    parser.add_argument(
        "--sampling",
        choices=["without_replacement", "with_replacement"],
        default="without_replacement",
    )
    parser.add_argument("--pi-lr", type=float, default=1e-3)
    parser.add_argument("--pi-decay", type=float, default=1e-2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    torch.manual_seed(seed)
    device = torch.device("cuda")

    root = Path(__file__).resolve().parent
    run_dir = root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f_ALSO")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(optimizer="ALSO", n_groups=100, split_seed=42,
                  mode="optimistic", alpha=1.0,
                  upstream_commit="bc4a5d8de99de382fc87632da46357280b16e6b9")
    config["data_root"] = str(args.data_root or root / "data")
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("ALSO config:", config)

    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_source = datasets.CIFAR100(
        root=config["data_root"],
        train=True,
        download=True,
        transform=train_transform,
    )
    val_source = datasets.CIFAR100(
        root=config["data_root"],
        train=True,
        download=False,
        transform=val_transform,
    )

    targets = torch.tensor(train_source.targets)
    split_generator = torch.Generator().manual_seed(42)
    train_indices = []
    val_indices = []

    for class_id in range(100):
        indices = torch.where(targets == class_id)[0]
        order = torch.randperm(len(indices), generator=split_generator)
        indices = indices[order]

        val_indices.extend(indices[:20].tolist())
        train_indices.extend(indices[20:].tolist())

    assert len(set(train_indices) & set(val_indices)) == 0

    torch.save(
        {"train": train_indices, "validation": val_indices},
        run_dir / "split.pt",
    )

    train_dataset = Subset(train_source, train_indices)

    sampler = RandomSampler(
        train_dataset,
        replacement=(args.sampling == "with_replacement"),
        num_samples=len(train_dataset),
        generator=torch.Generator().manual_seed(seed + 1),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed + 2),
        num_workers=0,
    )

    print("Сэмплирование:", args.sampling)
    val_loader = DataLoader(
        Subset(val_source, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    model = ResNet32(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    pi_uniform = torch.full((100,), 1 / 100, device=device)
    optimizer = ALSO(
        model.parameters(), n_groups=100, batch_size=batch_size,
        mode="optimistic", alpha=1.0,
        lr=learning_rate, weight_decay=weight_decay,
        pi_lr=args.pi_lr, pi_decay=args.pi_decay,
        pi_init=pi_uniform, pi_reg=pi_uniform,
    )
    torch.save(optimizer.pi.cpu(), run_dir / "pi_initial.pt")

    print(f"Train: {len(train_indices)}, validation: {len(val_indices)}")
    print(f"Батчей в обучающей эпохе: {len(train_loader)}")
    print(f"Результаты: {run_dir}")

    with (run_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "seconds",
        ])

        best_val_accuracy = -1.0

        for epoch in range(1, epochs + 1):
            start = time.perf_counter()

            # learm
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0

            for step, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)

                batch_log = {}
                pi_before = optimizer.pi.detach().clone()
                # Match the normalization to the actual batch length.
                optimizer.loss_scale = 100 / labels.numel()

                def closure(weights, scale):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    raw_losses = nn.functional.cross_entropy(
                        logits, labels, reduction="none"
                    )
                    assert weights.shape == raw_losses.shape == labels.shape
                    scaled_losses = scale * raw_losses
                    weighted_loss = (weights * scaled_losses).sum()
                    if epoch == 1 and step == 1:
                        torch.testing.assert_close(weighted_loss, raw_losses.mean())
                        print("CHECK: uniform weighted loss equals mean CE")
                    if not torch.isfinite(weighted_loss):
                        raise RuntimeError("Non-finite weighted loss")
                    weighted_loss.backward()
                    if args.smoke_test:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                assert torch.isfinite(parameter.grad).all()
                    batch_log["logits"] = logits.detach()
                    batch_log["loss"] = raw_losses.detach().mean()
                    return scaled_losses.detach(), raw_losses.detach().mean().item()

                closure.device = device
                optimizer.step(closure=closure, groups_indexes=labels)
                pi = optimizer.pi.detach()
                if not torch.isfinite(pi).all() or not (pi > 0).all():
                    raise RuntimeError("Invalid group weights")
                torch.testing.assert_close(pi.sum(), pi.new_tensor(1.0))
                logits, loss = batch_log["logits"], batch_log["loss"]
                if args.smoke_test:
                    assert all(torch.isfinite(p).all() for p in model.parameters())
                    print(
                        f"CHECK step={step}: CE={loss.item():.6f}, "
                        f"pi_sum={pi.sum().item():.6f}, "
                        f"pi_min={pi.min().item():.8f}, "
                        f"pi_max={pi.max().item():.8f}, "
                        f"pi_change={(pi-pi_before).abs().max().item():.8f}"
                    )
                    if step == 2:
                        print("SMOKE TEST PASSED (2 batches; no full epoch)")
                        return

                count = labels.size(0)
                train_loss_sum += loss.item() * count
                train_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                train_count += count

                if step % 100 == 0:
                    print(
                        f"  Эпоха {epoch}: "
                        f"батч {step}/{len(train_loader)}"
                    )

            # validaciya
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_count = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    count = labels.size(0)
                    val_loss_sum += loss.item() * count
                    val_correct += (
                        logits.argmax(dim=1) == labels
                    ).sum().item()
                    val_count += count

            torch.save(optimizer.pi.detach().cpu(), run_dir / f"pi_epoch_{epoch:03d}.pt")
            pi = optimizer.pi.detach()
            print(f"  pi: min={pi.min().item():.6f}, max={pi.max().item():.6f}, "
                  f"entropy={-(pi * pi.log()).sum().item():.4f}")
            train_loss = train_loss_sum / train_count
            train_accuracy = train_correct / train_count
            val_loss = val_loss_sum / val_count
            val_accuracy = val_correct / val_count
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_accuracy": val_accuracy,
                        "pi": optimizer.pi.detach().cpu(),
                        "config": config,
                        "checkpoint_purpose": "evaluation_only_not_training_resume",
                    },
                    run_dir / "model_best.pt",
                )

                print(
                    f"  Сохранена лучшая модель: эпоха {epoch}, "
                    f"val accuracy {val_accuracy:.1%}"
                )
            seconds = time.perf_counter() - start

            writer.writerow([
                epoch, train_loss, train_accuracy,
                val_loss, val_accuracy, seconds,
            ])
            file.flush()

            print(
                f"Эпоха {epoch}/{epochs} | "
                f"train loss {train_loss:.3f}, "
                f"acc {train_accuracy:.1%} | "
                f"val loss {val_loss:.3f}, "
                f"acc {val_accuracy:.1%} | "
                f"{seconds:.1f} сек."
            )

    torch.save({"model_state_dict": model.state_dict(),
                "pi": optimizer.pi.detach().cpu(), "config": config,
                "epoch": epochs, "checkpoint_purpose": "evaluation_only_not_training_resume"},
               run_dir / "model_last.pt")
    print("Пробный запуск завершён.")


if __name__ == "__main__":
    main()