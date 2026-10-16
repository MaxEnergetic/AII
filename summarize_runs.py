from pathlib import Path

import torch


root = Path(__file__).resolve().parent

for name in [
    "20261016_011345",
    "20261016_095031",
    "20261016_100255",
    "20261016_101847",
]:
    path = root / "runs" / name / "model_best.pt"

    if not path.exists():
        print(f"{name}: model_best.pt не найден")
        continue

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    print(f"\nЗапуск: {name}")
    print("Настройки:", checkpoint["config"])
    print("Лучшая эпоха:", checkpoint["epoch"])
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2%}")