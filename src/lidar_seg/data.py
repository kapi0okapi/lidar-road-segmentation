"""Training samples for RandLA-Net: fixed-size neighbourhoods ("spheres") cut from large tiles,
with the multi-level neighbour / pooling / upsampling indices the network needs.

The indices are computed on the CPU in DataLoader workers (scipy cKDTree), so the network
itself needs no custom CUDA operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from lidar_seg.io import IGNORE


@dataclass
class Tile:
    name: str
    xyz: np.ndarray  # (N, 3) float32, offset-free metres
    features: np.ndarray  # (N, C) float32
    labels: np.ndarray  # (N,) uint8, IGNORE for unlabelled
    tree: cKDTree

    def __len__(self) -> int:
        return len(self.xyz)


def tile_features(data, feature_names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in feature_names:
        if name == "rgb":
            columns.append(data["rgb"].astype(np.float32) / 255.0)
        elif name == "intensity":
            intensity = data["intensity"].astype(np.float32)
            # a few scans have missing intensities (about 0.5 % of Toronto-3D L003): they get the
            # tile's median, so the percentile normalisation stays finite
            intensity = np.where(np.isfinite(intensity), intensity, np.nanmedian(intensity))
            columns.append((intensity / (np.percentile(intensity, 99) + 1e-6)).clip(0, 2)[:, None])
        else:
            raise ValueError(f"unknown feature '{name}'")
    return np.concatenate(columns, axis=1) if columns else np.zeros((len(data["xyz"]), 0), np.float32)


def load_tile(path: Path, feature_names: tuple[str, ...]) -> Tile:
    data = np.load(path)
    xyz = data["xyz"].astype(np.float32)
    labels = data["labels"] if "labels" in data else np.full(len(xyz), IGNORE, np.uint8)
    return Tile(path.stem, xyz, tile_features(data, feature_names), labels, cKDTree(xyz))


def multilevel_indices(xyz: np.ndarray, k: int, ratios: tuple[int, ...]) -> dict[str, list[np.ndarray]]:
    """Per level: point coordinates, k-NN indices, pooling indices (k-NN of the kept points) and
    upsampling indices (nearest kept point for every point). Points must be in random order:
    keeping the first N/ratio of them is RandLA-Net's random sampling."""
    out = {"xyz": [], "neighbors": [], "pool": [], "up": []}
    current = xyz
    for ratio in ratios:
        neighbors = cKDTree(current).query(current, k=k)[1].astype(np.int64)
        n_sub = max(1, len(current) // ratio)
        sub = current[:n_sub]
        out["xyz"].append(current)
        out["neighbors"].append(neighbors)
        out["pool"].append(neighbors[:n_sub])
        out["up"].append(cKDTree(sub).query(current, k=1)[1].astype(np.int64))
        current = sub
    return out


class SphereDataset(Dataset):
    """``length`` samples per epoch, each the ``num_points`` nearest points around a centre.

    Training centres are class-balanced (pick a class, then a point of it) so rare classes such
    as poles and fences are seen often; evaluation uses fixed, seed-determined centres."""

    def __init__(
        self,
        tiles: list[Tile],
        num_points: int,
        k: int,
        ratios: tuple[int, ...],
        length: int,
        augment: bool,
        class_balanced: bool,
        seed: int = 0,
        num_classes: int = 8,
    ):
        self.tiles, self.num_points, self.k, self.ratios = tiles, num_points, k, ratios
        self.length, self.augment, self.class_balanced, self.seed = length, augment, class_balanced, seed
        self.epoch = 0
        self.class_index = [
            {c: np.flatnonzero(t.labels == c) for c in range(num_classes) if (t.labels == c).any()}
            for t in tiles
        ]
        sizes = np.array([len(t) for t in tiles], dtype=float)
        self.tile_prob = sizes / sizes.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        t = int(rng.choice(len(self.tiles), p=self.tile_prob))
        tile = self.tiles[t]
        if self.class_balanced and self.class_index[t]:
            cls = rng.choice(list(self.class_index[t]))
            center = tile.xyz[rng.choice(self.class_index[t][cls])]
        else:
            center = tile.xyz[rng.integers(len(tile))]
        center = center + rng.normal(0, 0.1, 3).astype(np.float32)
        return make_sample(tile, center, self.num_points, self.k, self.ratios, rng if self.augment else None)


def make_sample(
    tile: Tile,
    center: np.ndarray,
    num_points: int,
    k: int,
    ratios: tuple[int, ...],
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    k_query = min(num_points, len(tile))
    idx = tile.tree.query(center, k=k_query)[1]
    if k_query < num_points:  # small tile: pad by repeating points
        idx = np.concatenate([idx, np.random.default_rng(0).choice(idx, num_points - k_query)])
    shuffle = (rng or np.random.default_rng(0)).permutation(num_points)
    idx = idx[shuffle]
    xyz = tile.xyz[idx] - center.astype(np.float32)
    if rng is not None:
        xyz = augment(xyz, rng)
    levels = multilevel_indices(xyz, k, ratios)
    sample = {
        "features": torch.from_numpy(np.concatenate([xyz, tile.features[idx]], axis=1).astype(np.float32)),
        "labels": torch.from_numpy(tile.labels[idx].astype(np.int64)),
        "point_idx": torch.from_numpy(idx.astype(np.int64)),
    }
    for key, values in levels.items():
        for level, array in enumerate(values):
            sample[f"{key}_{level}"] = torch.from_numpy(
                array.astype(np.float32 if key == "xyz" else np.int64)
            )
    return sample


def augment(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi)  # rotation about the vertical axis only: gravity stays down
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    xyz = xyz @ rot.T * rng.uniform(0.9, 1.1)
    if rng.random() < 0.5:
        xyz[:, 0] *= -1
    return (xyz + rng.normal(0, 0.01, xyz.shape)).astype(np.float32)


def split_levels(batch: dict[str, torch.Tensor], num_levels: int) -> dict[str, list[torch.Tensor]]:
    return {
        key: [batch[f"{key}_{i}"] for i in range(num_levels)] for key in ("xyz", "neighbors", "pool", "up")
    }
