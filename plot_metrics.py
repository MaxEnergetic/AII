import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    run_dir = root / "runs" / "20261016_002308"

    with (run_dir / "metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    # В CSV точность записана долей: 0.55 означает 55%.
    train_acc = [100 * float(row["train_accuracy"]) for row in rows]
    val_acc = [100 * float(row["val_accuracy"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Cross-entropy")

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()

    output = run_dir / "learning_curves.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    best = max(rows, key=lambda row: float(row["val_accuracy"]))
    print("Графики сохранены:", output)
    print(
        f"Лучшая эпоха: {best['epoch']}, "
        f"validation accuracy: {float(best['val_accuracy']):.2%}"
    )


if __name__ == "__main__":
    main()