"""CIFAR-100 / ResNet34: original ALSO or ALSO with adaptive group subsampling."""

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
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34

from also import ALSO
from sampler import NoReplacementSampler, WeightAdaptiveSampler

N_GROUPS = 100


def build_model():
    model = resnet34(weights=None, num_classes=N_GROUPS)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def stratified_split(targets):
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [], []
    for group in range(N_GROUPS):
        indices = torch.where(targets == group)[0]
        if len(indices) <= 20:
            raise ValueError(f"Group {group} needs more than 20 examples")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:20].tolist())
        train_indices.extend(indices[20:].tolist())
    return train_indices, val_indices


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def learning_rate_for_epoch(args, epoch):
    """ALSO uses optimizer.lr, so a torch scheduler alone would not change it."""
    drops = sum(epoch > milestone for milestone in args.milestones) if args.scheduler == "multistep" else 0
    return args.lr * args.lr_gamma ** drops


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
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--sampling", choices=["adaptive", "without_replacement", "with_replacement"],
                        default="adaptive")
    parser.add_argument("--pi-lr", type=float, default=1e-3)
    parser.add_argument("--pi-decay", type=float, default=1e-2)
    parser.add_argument("--mode", choices=["optimistic", "descent-ascent"], default="optimistic")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-group-fraction", type=float, default=0.1)
    parser.add_argument("--min-per-group", type=int, default=16)
    parser.add_argument("--weight-ema", type=float, default=0.0,
                        help="Weight smoothing: 0 disables it; larger values react more slowly")
    parser.add_argument("--compensation", type=float, default=0.25,
                        help="Model-weight bonus = compensation * discarded group fraction (0..1)")
    parser.add_argument("--pi-correction", choices=["frequency", "none"], default="frequency",
                        help="Correct the pi loss signal for changed group observation frequency")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Stop after two training batches")
    parser.add_argument("--no-save-model", action="store_true", help="Save metrics and sampling state only")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative")
    if args.crop_padding < 0 or args.warmup_epochs < 0 or args.min_per_group < 1:
        parser.error("Invalid crop-padding, warmup-epochs or min-per-group")
    if not 0 < args.min_group_fraction <= 1 or not 0 <= args.weight_ema < 1:
        parser.error("min-group-fraction must be in (0,1]; weight-ema in [0,1)")