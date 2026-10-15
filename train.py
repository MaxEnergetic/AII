import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ResNet32


def main():
    seed = 42
    batch_size = 128
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4

    torch.manual_seed(seed)
    device = torch.device("cuda")

    root = Path(__file__).resolve().parent
    run_dir = root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

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
        root=str(root / "data"),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_source = datasets.CIFAR100(
        root=str(root / "data"),
        train=True,
        download=False,
        transform=val_transform,
    )

    targets = torch.tensor(train_source.targets)
    split_generator = torch.Generator().manual_seed(seed)
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

    train_loader = DataLoader(
        Subset(train_source, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_source, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    model = ResNet32(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

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

            train_loss = train_loss_sum / train_count
            train_accuracy = train_correct / train_count
            val_loss = val_loss_sum / val_count
            val_accuracy = val_correct / val_count
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

    torch.save(model.state_dict(), run_dir / "model_last.pt")
    print("Пробный запуск завершён.")


if __name__ == "__main__":
    main()