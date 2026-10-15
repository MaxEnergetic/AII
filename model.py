import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Residual-блок: выход = основная ветка + короткий путь."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.extra_channels = out_channels - in_channels

    def forward(self, x):
        shortcut = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # При переходе между стадиями уменьшаем размер картинки.
        shortcut = shortcut[:, :, ::self.stride, ::self.stride]

        # Дополняем каналы нулями: shortcut без обучаемых параметров.
        if self.extra_channels:
            pad = self.extra_channels // 2
            shortcut = F.pad(shortcut, (0, 0, 0, 0, pad, pad))

        return F.relu(out + shortcut)


class ResNet32(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.stage1 = self.make_stage(16, 16, stride=1)
        self.stage2 = self.make_stage(16, 32, stride=2)
        self.stage3 = self.make_stage(32, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    @staticmethod
    def make_stage(in_channels, out_channels, stride):
        blocks = [BasicBlock(in_channels, out_channels, stride)]
        blocks += [BasicBlock(out_channels, out_channels) for _ in range(4)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)