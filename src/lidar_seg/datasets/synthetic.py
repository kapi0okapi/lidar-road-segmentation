"""Synthetic road-corridor scans with the Toronto-3D classes, for tests and quick local runs.

Guardrails are W-beam style rails on posts with a known rail height, which lets the barrier
measurement be checked against ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_seg.datasets.toronto3d import CLASSES

ROAD, MARKING, NATURAL, BUILDING, UTILITY, POLE, CAR, FENCE = range(len(CLASSES))


def _box(rng, n, lo, hi):
    return rng.uniform(lo, hi, (n, 3))


def road_scene(length: float = 60.0, rail_height: float = 0.75, density: float = 400.0, seed: int = 0):
    """Returns xyz (N, 3), rgb (N, 3) uint8, intensity (N,), labels (N,) and the rail top height
    above the road surface. The road runs along x with a gentle longitudinal slope."""
    rng = np.random.default_rng(seed)
    parts = []

    def ground(x, y):
        return 0.02 * x + 0.01 * np.sin(y)  # 2 % grade, slight camber

    def add(xyz, label, color, intensity):
        n = len(xyz)
        rgb = np.clip(np.array(color) + rng.normal(0, 12, (n, 3)), 0, 255)
        parts.append(
            (xyz, rgb.astype(np.uint8), np.full(n, intensity, np.float32) + rng.normal(0, 3, n), label)
        )

    n = int(length * 14 * density / 10)
    road = _box(rng, n, (0, -7, 0), (length, 7, 0))
    road[:, 2] = ground(road[:, 0], road[:, 1]) + rng.normal(0, 0.01, n)
    marking = (np.abs(road[:, 1]) < 0.08) & ((road[:, 0] % 6) < 3)
    add(road[~marking], ROAD, (70, 70, 75), 20)
    add(road[marking], MARKING, (230, 230, 230), 80)

    for side in (-1, 1):  # guardrails along both road edges
        y = side * 7.4
        m = int(length * density / 4)
        rail = np.column_stack([rng.uniform(0, length, m), y + rng.normal(0, 0.02, m), np.zeros(m)])
        rail[:, 2] = ground(rail[:, 0], y) + rng.uniform(rail_height - 0.3, rail_height, m)  # 0.3 m tall beam
        add(rail, FENCE, (180, 180, 185), 60)
        for x in np.arange(1.0, length, 2.0):  # posts
            k = 40
            post = np.column_stack(
                [
                    x + rng.normal(0, 0.03, k),
                    y + side * 0.15 + rng.normal(0, 0.03, k),
                    ground(x, y) + rng.uniform(0, rail_height - 0.05, k),
                ]
            )
            add(post, FENCE, (150, 150, 150), 50)

    for x in np.arange(5.0, length, 15.0):  # light poles with a wire on top
        k = 300
        pole = np.column_stack(
            [x + rng.normal(0, 0.05, k), -9.0 + rng.normal(0, 0.05, k), rng.uniform(0, 8, k)]
        )
        pole[:, 2] += ground(x, -9.0)
        add(pole, POLE, (110, 110, 100), 40)
    wire = np.column_stack([rng.uniform(0, length, 600), np.full(600, -9.0), np.zeros(600)])
    wire[:, 2] = ground(wire[:, 0], -9.0) + 8.0 + 0.3 * np.cos(wire[:, 0] / 15.0 * 2 * np.pi)
    add(wire, UTILITY, (40, 40, 40), 10)

    for x in rng.uniform(5, length - 5, 4):  # parked and moving cars
        y = rng.choice([-3.5, 3.5])
        car = _box(rng, 1500, (x - 2.2, y - 0.9, 0.3), (x + 2.2, y + 0.9, 1.5))
        car[:, 2] += ground(x, y)
        add(car, CAR, rng.integers(30, 220, 3), 35)

    trees = _box(rng, 4000, (0, 10, 0), (length, 16, 0))
    trees[:, 2] = ground(trees[:, 0], trees[:, 1]) + rng.uniform(0, 7, len(trees))
    add(trees, NATURAL, (60, 120, 50), 25)
    wall = _box(rng, 5000, (0, -20, 0), (length, -19.8, 12))
    wall[:, 2] += ground(wall[:, 0], wall[:, 1])
    add(wall, BUILDING, (170, 140, 110), 45)

    xyz = np.concatenate([p[0] for p in parts])
    rgb = np.concatenate([p[1] for p in parts])
    intensity = np.concatenate([p[2] for p in parts])
    labels = np.concatenate([np.full(len(p[0]), p[3], np.uint8) for p in parts])
    order = rng.permutation(len(xyz))
    return xyz[order], rgb[order], intensity[order], labels[order]


def write_prepared(out_dir: Path, names=("S001", "S002"), **kwargs) -> Path:
    """Writes tiles in the same .npz layout as the Toronto-3D preprocessing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        xyz, rgb, intensity, labels = road_scene(seed=i, **kwargs)
        np.savez(
            out_dir / f"{name}.npz",
            xyz=xyz.astype(np.float32),
            rgb=rgb,
            intensity=intensity,
            labels=labels,
            proj=np.arange(len(xyz), dtype=np.int32),
            full_labels=labels,
        )
    return out_dir
