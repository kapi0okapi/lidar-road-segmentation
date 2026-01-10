"""RandLA-Net (Hu et al., CVPR 2020) in plain PyTorch.

Large point clouds are processed by random sampling (cheap, unlike farthest-point sampling)
combined with local feature aggregation that keeps geometric detail: a relative position
encoding of each point's k nearest neighbours and attentive pooling over them.

Tensors are (B, C, N, 1) so 1x1 convolutions act as shared MLPs; neighbourhoods are (B, C, N, K).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gather(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features (B, C, N), idx (B, M, K) -> (B, C, M, K)."""
    b, c, _ = features.shape
    m, k = idx.shape[1:]
    return features.gather(2, idx.reshape(b, 1, m * k).expand(-1, c, -1)).reshape(b, c, m, k)


class SharedMLP(nn.Sequential):
    def __init__(self, cin: int, cout: int, activation: bool = True):
        layers = [nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        if activation:
            layers.append(nn.LeakyReLU(0.2))
        super().__init__(*layers)


class RelativePositionEncoding(nn.Module):
    """[distance, p_i - p_k, p_i, p_k] for every neighbour, mapped to ``d`` channels."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = SharedMLP(10, d)

    def forward(self, xyz: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        points = xyz.transpose(1, 2)  # (B, 3, N)
        neighbor_xyz = gather(points, neighbors)
        center = points.unsqueeze(-1).expand_as(neighbor_xyz)
        relative = center - neighbor_xyz
        distance = relative.norm(dim=1, keepdim=True)
        return self.mlp(torch.cat([distance, relative, center, neighbor_xyz], dim=1))


class AttentivePooling(nn.Module):
    """Learned softmax weights over the K neighbours instead of max/mean pooling."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.score = nn.Conv2d(cin, cin, 1, bias=False)
        self.mlp = SharedMLP(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.score(x).softmax(dim=-1)
        return self.mlp((x * weights).sum(dim=-1, keepdim=True))


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        half = d_out // 2
        self.encode1 = RelativePositionEncoding(half)
        self.pool1 = AttentivePooling(d_out, half)
        self.encode2 = SharedMLP(half, half)
        self.pool2 = AttentivePooling(d_out, d_out)

    def forward(self, features: torch.Tensor, xyz: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        geometry = self.encode1(xyz, neighbors)
        x = torch.cat([gather(features.squeeze(-1), neighbors), geometry], dim=1)
        x = self.pool1(x)
        geometry = self.encode2(geometry)
        x = torch.cat([gather(x.squeeze(-1), neighbors), geometry], dim=1)
        return self.pool2(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp1 = SharedMLP(d_in, d_out // 2)
        self.lfa = LocalFeatureAggregation(d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out, activation=False)
        self.shortcut = SharedMLP(d_in, 2 * d_out, activation=False)

    def forward(self, features, xyz, neighbors):
        x = self.mlp2(self.lfa(self.mlp1(features), xyz, neighbors))
        return F.leaky_relu(x + self.shortcut(features), 0.2)


class RandLANet(nn.Module):
    def __init__(self, d_in: int, num_classes: int, d_out: tuple[int, ...] = (16, 64, 128, 256, 512)):
        super().__init__()
        self.fc0 = nn.Sequential(nn.Linear(d_in, 8), nn.LayerNorm(8), nn.LeakyReLU(0.2))
        self.encoder = nn.ModuleList()
        channels, skips = 8, []
        for d in d_out:
            self.encoder.append(DilatedResidualBlock(channels, d))
            channels = 2 * d
            skips.append(channels)
        self.bottleneck = SharedMLP(channels, channels)
        skip_channels = [2 * d_out[0], *skips]  # full-resolution encoder output, then every sampled level
        self.decoder = nn.ModuleList()
        for j in range(len(d_out)):
            target = skip_channels[-j - 2]
            self.decoder.append(SharedMLP(channels + target, target))
            channels = target
        self.head = nn.Sequential(
            SharedMLP(channels, 64), SharedMLP(64, 32), nn.Dropout(0.5), nn.Conv2d(32, num_classes, 1)
        )

    @staticmethod
    def downsample(features: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        return gather(features.squeeze(-1), pool).max(dim=-1, keepdim=True)[0]

    @staticmethod
    def upsample(features: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return gather(features.squeeze(-1), up.unsqueeze(-1))

    def forward(self, features: torch.Tensor, levels: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        """features (B, N, d_in) -> logits (B, num_classes, N)."""
        x = self.fc0(features).transpose(1, 2).unsqueeze(-1)  # (B, 8, N, 1)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x, levels["xyz"][i], levels["neighbors"][i])
            if i == 0:
                skips.append(x)
            x = self.downsample(x, levels["pool"][i])
            skips.append(x)
        x = self.bottleneck(x)
        for j, block in enumerate(self.decoder):
            x = self.upsample(x, levels["up"][-j - 1])
            x = block(torch.cat([skips[-j - 2], x], dim=1))
        return self.head(x).squeeze(-1)
