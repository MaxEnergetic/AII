from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet32


def main():
    data_dir = Path(__file__).resolve().parent / "data"

    dataset = datasets.CIFAR100(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Количество картинок:", len(dataset))
    print("Количество классов:", len(dataset.classes))
    print("Размер батча картинок:", images.shape)
    print("Размер батча меток:", labels.shape)

    images = images.to("cuda")
    labels = labels.to("cuda")

    print("Картинки находятся на:", images.device)
    print("Метки находятся на:", labels.device)

    # Здесь начинается новый код — модель и один шаг обучения.
    model = ResNet32(num_classes=100).to("cuda")
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Используем одни и те же 128 картинок на всех 100 шагах.
    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

            print(
                f"Шаг {step:3d}/100 | "
                f"loss: {loss.item():.4f} | "
                f"accuracy: {accuracy * 100:.1f}%"
            )



if __name__ == "__main__":
    main()