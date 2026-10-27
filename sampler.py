"""Samplers return positions in the training Subset, never CIFAR source IDs."""

import math

import torch
from torch.utils.data import Sampler


class NoReplacementSampler(Sampler):
    def __init__(self, dataset, seed=42):
        self.size = len(dataset)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return iter(torch.randperm(self.size, generator=self.generator).tolist())

    def __len__(self):
        return self.size


class WeightAdaptiveSampler(Sampler):
    """Keep a random fraction of each group, proportional to its ALSO weight.

    Call set_epoch BEFORE iterating the DataLoader. Counts stay fixed throughout
    that epoch. Every epoch draws a fresh subset without replacement; skipped
    items remain eligible next epoch. No extra items fill the freed positions.

    The reference is the initial normalized pi, not the previous active count.
    Thus increasing pi can restore a group even after it hits the minimum.
    """

    def __init__(
        self, group_ids, reference_weights, seed=42, warmup_epochs=2,
        min_fraction=0.1, min_per_group=16, weight_ema=0.5,
        compensation=0.25,
    ):
        ids = torch.as_tensor(group_ids, device="cpu")
        if ids.ndim != 1 or ids.numel() == 0 or ids.is_floating_point():
            raise ValueError("group_ids must be a nonempty vector of integer IDs")
        self.group_ids = ids.to(torch.long).clone()
        self.reference_weights = self._normalize_weights(reference_weights)
        self.n_groups = self.reference_weights.numel()
        if self.group_ids.min() < 0 or self.group_ids.max() >= self.n_groups:
            raise ValueError("group_ids must index reference_weights")
        if warmup_epochs < 0 or int(warmup_epochs) != warmup_epochs:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        if not 0 < min_fraction <= 1:
            raise ValueError("min_fraction must be in (0, 1]")
        if min_per_group < 1 or int(min_per_group) != min_per_group:
            raise ValueError("min_per_group must be a positive integer")
        if not 0 <= weight_ema < 1:
            raise ValueError("weight_ema must be in [0, 1)")
        if not math.isfinite(compensation) or not 0 <= compensation <= 1:
            raise ValueError("compensation must be in [0, 1]")

        self.members = [torch.where(self.group_ids == k)[0]
                        for k in range(self.n_groups)]
        self.group_sizes = torch.tensor([len(x) for x in self.members])
        if (self.group_sizes == 0).any():
            raise ValueError("Each group must contain at least one training item")
        self.full_size = len(self.group_ids)
        self.seed = seed
        self.warmup_epochs = int(warmup_epochs)
        self.min_fraction = min_fraction
        self.min_per_group = int(min_per_group)
        self.weight_ema = weight_ema
        self.compensation = compensation
        self.smoothed_weights = self.reference_weights.clone()
        self.active_counts = self.group_sizes.clone()
        self.active_indices = torch.empty(0, dtype=torch.long)
        self.epoch = 0

    @staticmethod
    def _normalize_weights(weights):
        weights = torch.as_tensor(weights, device="cpu", dtype=torch.float64).detach()
        if (weights.ndim != 1 or weights.numel() == 0
                or not torch.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError("Weights must be a finite, positive vector")
        total = weights.sum()
        if not torch.isfinite(total):
            raise ValueError("Weight sum must be finite")
        return weights / total

    def set_epoch(self, epoch, weights):
        if epoch != self.epoch + 1:
            raise ValueError("Call set_epoch once per epoch, starting at 1")
        weights = self._normalize_weights(weights)
        if weights.shape != self.reference_weights.shape:
            raise ValueError("Weights must have one entry per group")
        self.smoothed_weights = (
            self.weight_ema * self.smoothed_weights + (1 - self.weight_ema) * weights
        )
        if epoch <= self.warmup_epochs:
            self.active_counts = self.group_sizes.clone()
        else:
            fraction = (self.smoothed_weights / self.reference_weights).clamp(
                min=self.min_fraction, max=1.0,
            )
            requested = torch.ceil(self.group_sizes * fraction).to(torch.long)
            minimum = torch.minimum(
                self.group_sizes, torch.full_like(self.group_sizes, self.min_per_group),
            )
            self.active_counts = torch.minimum(
                self.group_sizes, torch.maximum(requested, minimum),
            )

        # Separate generator: deterministic selection does not consume model RNG.
        generator = torch.Generator().manual_seed(self.seed + epoch)
        chosen = [members[torch.randperm(len(members), generator=generator)[:count]]
                  for members, count in zip(self.members, self.active_counts.tolist())]
        active = torch.cat(chosen)
        self.active_indices = active[torch.randperm(len(active), generator=generator)]
        self.epoch = epoch

    @property
    def retention(self):
        return self.active_counts.to(torch.float64) / self.group_sizes

    @property
    def overall_retention(self):
        return int(self.active_counts.sum()) / self.full_size

    @property
    def model_weight_multipliers(self):
        # Small, bounded bonus; NOT full inverse-probability compensation.
        # Keep 20% with compensation=.25 -> multiply training weight by 1.20.
        return 1 + self.compensation * (1 - self.retention)

    @property
    def pi_loss_multipliers(self):
        # Original group frequency / selected group frequency. This corrects
        # the pi signal only, not the model gradient. Exact frequency correction
        # for fixed losses in expectation over the random subset/permutation;
        # adaptive without-replacement training is not conditionally unbiased.
        return self.overall_retention / self.retention

    def __iter__(self):
        if self.epoch == 0:
            raise RuntimeError("Call set_epoch before iterating WeightAdaptiveSampler")
        return iter(self.active_indices.tolist())

    def __len__(self):
        return int(self.active_counts.sum())

