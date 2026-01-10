"""Toronto-3D: ~1 km of urban road scanned by a mobile laser scanner (Teledyne Optech Maverick),
78.3 M points in four tiles with per-point labels.

Download (CC BY-NC 4.0): https://github.com/WeikaiTan/Toronto-3D, files L001.ply ... L004.ply.
Standard split: train on L001, L003, L004 and test on L002.

    python -m lidar_seg.datasets.toronto3d --raw data/toronto3d --out data/toronto3d_prepared --voxel 0.06
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from lidar_seg.io import IGNORE, grid_subsample, read_ply

log = logging.getLogger(__name__)

CLASSES = ("road", "road_marking", "natural", "building", "utility_line", "pole", "car", "fence")
UTM_OFFSET = np.array([627285.0, 4841948.0, 0.0])  # subtracted to keep float32 precision
TILES = {"train": ("L001", "L003", "L004"), "test": ("L002",)}


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """File labels: 0 unclassified, 1..8 classes -> 0..7, unclassified -> IGNORE."""
    out = raw.astype(np.int16) - 1
    out[(out < 0) | (out >= len(CLASSES))] = IGNORE
    return out.astype(np.uint8)


def prepare_tile(ply: Path, out_dir: Path, voxel: float) -> dict:
    cloud = read_ply(ply)
    cloud.xyz -= UTM_OFFSET
    cloud.labels = remap_labels(cloud.labels)
    sub, inverse = grid_subsample(cloud, voxel, len(CLASSES))
    np.savez(
        out_dir / f"{ply.stem}.npz",
        xyz=sub.xyz.astype(np.float32),
        rgb=sub.rgb,
        intensity=sub.intensity,
        labels=sub.labels,
        proj=inverse.astype(np.int32),  # full-resolution point -> voxel point
        full_labels=cloud.labels,
    )
    counts = np.bincount(sub.labels[sub.labels != IGNORE], minlength=len(CLASSES))
    return {"tile": ply.stem, "points": len(cloud), "subsampled": len(sub), "class_counts": counts.tolist()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, required=True, help="folder with L001.ply ... L004.ply")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.06)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    stats = []
    for name in sorted(n for tiles in TILES.values() for n in tiles):
        info = prepare_tile(args.raw / f"{name}.ply", args.out, args.voxel)
        log.info("%s: %d -> %d points", name, info["points"], info["subsampled"])
        stats.append(info)
    train_counts = np.sum([s["class_counts"] for s in stats if s["tile"] in TILES["train"]], axis=0)
    meta = {
        "classes": CLASSES,
        "voxel": args.voxel,
        "splits": TILES,
        "tiles": stats,
        "train_class_counts": train_counts.tolist(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
