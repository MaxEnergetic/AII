from pathlib import Path
import torch

root = Path(__file__).resolve().parent / "runs"
names = [
    "20261016_100255",  # seed 42
    "20261016_105716",  # seed 43
    "20261016_110805",  # seed 44
]

reference = torch.load(
    root / names[0] / "split.pt", weights_only=True
)

for name in names:
    split = torch.load(root / name / "split.pt", weights_only=True)

    same_train = set(split["train"]) == set(reference["train"])
    same_val = set(split["validation"]) == set(reference["validation"])

    print(f"{name}: train совпадает={same_train}, validation совпадает={same_val}")