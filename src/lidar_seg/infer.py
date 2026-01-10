"""Segment whole tiles, or any LAS/LAZ/PLY file, with overlapping neighbourhoods and voting.

Every point is covered by several neighbourhoods. Each new neighbourhood is centred on the
least-covered point, and the per-point class probabilities are averaged over all predictions
(RandLA-Net's test procedure). Predictions on the voxel-subsampled cloud are projected back to
every original point.

    lidar-infer --checkpoint runs/randla_toronto3d/best.pt --tile data/toronto3d_prepared/L002.npz
    lidar-infer --checkpoint best.pt --input scan.laz --out scan_classified.laz
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from lidar_seg.config import TrainConfig
from lidar_seg.data import Tile, load_tile, make_sample, split_levels, tile_features
from lidar_seg.io import PointCloud, grid_subsample, read_points, write_las
from lidar_seg.metrics import confusion, scores
from lidar_seg.model import RandLANet
from lidar_seg.train import build_model
from lidar_seg.utils.config import from_dict
from lidar_seg.utils.runtime import Precision, load_checkpoint, select_device

log = logging.getLogger("lidar_seg.infer")


def load_model(checkpoint: Path, device: torch.device) -> tuple[RandLANet, TrainConfig]:
    state = load_checkpoint(checkpoint, device)
    cfg = from_dict(TrainConfig, state["config"])
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_tile(
    model: RandLANet,
    tile: Tile,
    cfg: TrainConfig,
    device: torch.device,
    min_coverage: float = 1.0,
    batch_size: int = 8,
    smoothing: float = 0.95,
) -> np.ndarray:
    """Class probabilities (N, C) for every point of the tile."""
    d = cfg.data
    rng = np.random.default_rng(0)
    coverage = rng.random(len(tile)) * 1e-3
    probs = np.zeros((len(tile), len(d.classes)), np.float32)
    precision = Precision(device)
    while coverage.min() < min_coverage:
        samples = []
        for _ in range(batch_size):
            center = tile.xyz[int(np.argmin(coverage))]
            sample = make_sample(tile, center, d.num_points, d.k, d.ratios)
            idx = sample["point_idx"].numpy()
            dist2 = ((tile.xyz[idx] - center) ** 2).sum(1)
            coverage[idx] += 1.0 - dist2 / dist2.max()  # points near the centre count as better covered
            samples.append(sample)
        batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in samples[0]}
        with precision.autocast():
            logits = model(batch["features"], split_levels(batch, len(d.ratios)))
        p = logits.float().softmax(1).transpose(1, 2).cpu().numpy()
        for i, sample in enumerate(samples):
            idx = sample["point_idx"].numpy()
            probs[idx] = smoothing * probs[idx] + (1 - smoothing) * p[i]
    return probs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tile", type=Path, nargs="+", help="prepared .npz tiles (with labels: scored)")
    group.add_argument("--input", type=Path, help="raw LAS/LAZ/PLY file")
    parser.add_argument("--out", type=Path, help="LAS/LAZ output with predicted classes")
    parser.add_argument("--voxel", type=float, default=0.06, help="--input only")
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = select_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    n = len(cfg.data.classes)

    if args.input:
        cloud = read_points(args.input)
        origin = cloud.xyz.min(0)
        cloud.xyz = cloud.xyz - origin
        sub, inverse = grid_subsample(cloud, args.voxel, n)
        data = {"xyz": sub.xyz, "rgb": sub.rgb, "intensity": sub.intensity}
        missing = [f for f in cfg.data.features if data.get(f) is None]
        if missing:
            raise SystemExit(f"the model needs features {missing}, which {args.input.name} does not have")
        tile = Tile(
            args.input.stem,
            sub.xyz.astype(np.float32),
            tile_features(data, cfg.data.features),
            np.full(len(sub), 255, np.uint8),
            cKDTree(sub.xyz.astype(np.float32)),
        )
        started = time.time()
        labels = predict_tile(model, tile, cfg, device, args.min_coverage).argmax(1)[inverse]
        log.info(
            "%d points in %.1fs (%.0fk points/s)",
            len(cloud),
            time.time() - started,
            len(cloud) / (time.time() - started) / 1e3,
        )
        cloud.xyz = cloud.xyz + origin
        write_las(args.out or args.input.with_name(args.input.stem + "_classified.laz"), cloud, labels)
        return

    report, total_cm = {}, np.zeros((n, n), np.int64)
    for path in args.tile:
        tile = load_tile(path, cfg.data.features)
        archive = np.load(path)
        started = time.time()
        pred_sub = predict_tile(model, tile, cfg, device, args.min_coverage).argmax(1)
        elapsed = time.time() - started
        np.save(
            args.checkpoint.with_name(f"{path.stem}_pred.npy"), pred_sub.astype(np.uint8)
        )  # for lidar-barriers
        pred_full = pred_sub[archive["proj"]]
        cm = confusion(pred_full, archive["full_labels"], n)
        total_cm += cm
        report[path.stem] = {
            **scores(cm, cfg.data.classes),
            "seconds": elapsed,
            "points_per_s": len(pred_full) / elapsed,
        }
        log.info(
            "%s: mIoU %.3f, %.0fk points/s",
            path.stem,
            report[path.stem]["miou"],
            report[path.stem]["points_per_s"] / 1e3,
        )
        if args.out:
            cloud = PointCloud(tile.xyz.astype(np.float64), labels=tile.labels)
            write_las(args.out.with_name(f"{path.stem}_pred.laz"), cloud, pred_sub)
    report["all"] = scores(total_cm, cfg.data.classes)
    out = args.checkpoint.with_name("test_metrics.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
