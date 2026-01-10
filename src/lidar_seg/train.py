"""Train RandLA-Net on prepared tiles.

lidar-train --config configs/toronto3d.yaml
lidar-train --config configs/smoke.yaml --run-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lidar_seg.config import TrainConfig
from lidar_seg.data import SphereDataset, Tile, load_tile, split_levels
from lidar_seg.io import IGNORE
from lidar_seg.metrics import confusion, scores
from lidar_seg.model import RandLANet
from lidar_seg.utils.config import load_config, save_config
from lidar_seg.utils.runtime import (
    MetricLogger,
    Precision,
    StopSignal,
    cosine_with_warmup,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
    setup_logging,
)

log = logging.getLogger("lidar_seg.train")


def load_tiles(cfg: TrainConfig, names: tuple[str, ...]) -> list[Tile]:
    root = Path(os.path.expandvars(cfg.data.root))
    return [load_tile(root / f"{name}.npz", cfg.data.features) for name in names]


def build_model(cfg: TrainConfig) -> RandLANet:
    d_in = 3 + (3 if "rgb" in cfg.data.features else 0) + (1 if "intensity" in cfg.data.features else 0)
    return RandLANet(d_in, len(cfg.data.classes), tuple(cfg.model.d_out))


def class_weights(tiles: list[Tile], num_classes: int, power: float) -> torch.Tensor:
    labels = np.concatenate([t.labels for t in tiles])
    counts = np.bincount(labels[labels != IGNORE], minlength=num_classes).astype(float)
    freq = counts / counts.sum()
    weights = 1.0 / np.maximum(freq, 1e-6) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: RandLANet, loader: DataLoader, cfg: TrainConfig, device, precision: Precision) -> dict:
    """mIoU on sampled neighbourhoods (fast proxy during training; infer.py scores whole tiles)."""
    model.eval()
    n = len(cfg.data.classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for batch in loader:
        batch = to_device(batch, device)
        with precision.autocast():
            logits = model(batch["features"], split_levels(batch, len(cfg.data.ratios)))
        cm += confusion(logits.argmax(1).cpu().numpy().ravel(), batch["labels"].cpu().numpy().ravel(), n)
    return scores(cm, cfg.data.classes)


def train(cfg: TrainConfig, run_dir: Path, resume: bool = True) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    precision = Precision(device, cfg.amp)
    d = cfg.data

    train_tiles, val_tiles = load_tiles(cfg, d.train_tiles), load_tiles(cfg, d.val_tiles)
    log.info("train points %d, val points %d", sum(map(len, train_tiles)), sum(map(len, val_tiles)))
    train_set = SphereDataset(
        train_tiles,
        d.num_points,
        d.k,
        d.ratios,
        d.steps_per_epoch * d.batch_size,
        augment=True,
        class_balanced=True,
        seed=cfg.seed,
        num_classes=len(d.classes),
    )
    val_set = SphereDataset(
        val_tiles,
        d.num_points,
        d.k,
        d.ratios,
        d.val_samples,
        augment=False,
        class_balanced=False,
        seed=12345,
        num_classes=len(d.classes),
    )
    loader_kw = {
        "batch_size": d.batch_size,
        "num_workers": d.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_set, shuffle=False, persistent_workers=d.num_workers > 0, **loader_kw)

    model = build_model(cfg).to(device)
    weights = class_weights(train_tiles, len(d.classes), cfg.optim.class_weight_power).to(device)
    log.info("class weights %s", dict(zip(d.classes, weights.cpu().numpy().round(2), strict=True)))
    o = cfg.optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    scheduler = cosine_with_warmup(
        optimizer, int(o.warmup_epochs * d.steps_per_epoch), o.epochs * d.steps_per_epoch
    )
    logger = MetricLogger(run_dir)
    stop = StopSignal()

    start_epoch, best = 0, {"miou": -1.0}
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        precision.scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"], state["best"]
        log.info("resumed at epoch %d", start_epoch)

    def save_last(next_epoch: int) -> None:
        save_checkpoint(
            last_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=precision.scaler.state_dict(),
            epoch=next_epoch,
            best=best,
            config=dataclasses.asdict(cfg),
        )

    for epoch in range(start_epoch, o.epochs):
        train_set.set_epoch(epoch)
        model.train()
        started, total = time.time(), 0.0
        for batch in DataLoader(train_set, shuffle=False, drop_last=True, **loader_kw):
            batch = to_device(batch, device)
            with precision.autocast():
                logits = model(batch["features"], split_levels(batch, len(d.ratios)))
            loss = torch.nn.functional.cross_entropy(
                logits.float(), batch["labels"], weight=weights, ignore_index=IGNORE
            )
            precision.step(loss, optimizer, model, o.max_grad_norm)
            scheduler.step()
            total += loss.item()
            if stop:
                save_last(epoch)
                log.warning("stopped by signal during epoch %d, checkpoint saved", epoch)
                return best

        val = evaluate(model, val_loader, cfg, device, precision)
        points_per_s = d.steps_per_epoch * d.batch_size * d.num_points / (time.time() - started)
        logger.log(
            epoch,
            {
                "train/loss": total / d.steps_per_epoch,
                **{f"val/{k}": v for k, v in val.items()},
                "lr": scheduler.get_last_lr()[0],
                "train/points_per_s": points_per_s,
            },
        )
        log.info(
            "epoch %d/%d loss=%.4f val mIoU=%.3f OA=%.3f (%.0fs, %.0fk pts/s)",
            epoch + 1,
            o.epochs,
            total / d.steps_per_epoch,
            val["miou"],
            val["oa"],
            time.time() - started,
            points_per_s / 1e3,
        )
        if val["miou"] > best["miou"]:
            best = val
            save_checkpoint(
                run_dir / "best.pt",
                model=model.state_dict(),
                config=dataclasses.asdict(cfg),
                epoch=epoch,
                val=val,
            )
        save_last(epoch + 1)
    logger.close()
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-dir", type=str, help="default: $RUNS/<name> (or runs/<name>)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(TrainConfig, args.config, args.set)
    run_dir = Path(os.path.expandvars(args.run_dir or os.path.join(os.environ.get("RUNS", "runs"), cfg.name)))
    train(cfg, run_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
