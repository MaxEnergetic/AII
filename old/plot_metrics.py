"""Compare existing and adaptive ALSO runs by epochs, elapsed time and work."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run folders or metrics.csv paths")
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.png"))
    args = parser.parse_args(argv)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for run in args.runs:
        path = run if run.is_file() else run / "metrics.csv"
        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            print(f"Skipping empty metrics: {path}")
            continue
        label = path.parent.name
        epochs = [int(row["epoch"]) for row in rows]
        accuracy = [100 * float(row["val_accuracy"]) for row in rows]
        elapsed, total = [], 0.0
        for row in rows:
            total += float(row["seconds"])
            elapsed.append(float(row.get("cumulative_seconds") or total) / 60)
        axes[0, 0].plot(epochs, accuracy, label=label)
        axes[0, 1].plot(elapsed, accuracy, label=label)
        axes[0, 2].plot(epochs, [float(row["val_loss"]) for row in rows], label=label)
        if "train_examples" in rows[0]:
            axes[1, 0].plot(epochs, [int(row["train_examples"]) for row in rows], label=label)
            seen, total_seen = [], 0
            for row in rows:
                total_seen += int(row["train_examples"])
                seen.append(int(row.get("cumulative_train_examples") or total_seen) / 1000)
            axes[1, 1].plot(seen, accuracy, label=label)
        if "val_worst_group_accuracy" in rows[0]:
            axes[1, 2].plot(epochs, [100 * float(row["val_worst_group_accuracy"]) for row in rows], label=label)
        best = max(rows, key=lambda row: float(row["val_accuracy"]))
        print(f"{label}: best validation {float(best['val_accuracy']):.2%}, epoch {best['epoch']}; "
              f"recorded time {elapsed[-1]:.1f} min")
    labels = [
        ("Validation accuracy", "Epoch", "Accuracy (%)"),
        ("Validation accuracy vs elapsed time", "Minutes (recorded epochs)", "Accuracy (%)"),
        ("Validation loss", "Epoch", "Cross-entropy"),
        ("Training examples per epoch", "Epoch", "Examples processed"),
        ("Validation accuracy vs processed examples", "Thousands of training examples", "Accuracy (%)"),
        ("Worst group validation accuracy", "Epoch", "Accuracy (%)"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, labels):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "Not recorded in these runs", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print("Графики сохранены:", args.output)


if __name__ == "__main__":
    main()
