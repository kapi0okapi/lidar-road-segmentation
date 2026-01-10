"""Smoke test: prepare synthetic tiles, train one tiny epoch, run whole-tile inference."""

import json
from pathlib import Path

from lidar_seg import infer
from lidar_seg.config import TrainConfig
from lidar_seg.datasets.synthetic import write_prepared
from lidar_seg.train import train
from lidar_seg.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_train_and_infer(tmp_path):
    root = write_prepared(tmp_path / "tiles", length=20.0)
    cfg = load_config(TrainConfig, CONFIGS / "smoke.yaml", [f"data.root={root}"])
    best = train(cfg, tmp_path / "run")
    assert "miou" in best and (tmp_path / "run" / "best.pt").exists()
    infer.main(
        [
            "--checkpoint",
            str(tmp_path / "run" / "best.pt"),
            "--tile",
            str(root / "S002.npz"),
            "--device",
            "cpu",
            "--min-coverage",
            "0.2",
        ]
    )
    report = json.loads((tmp_path / "run" / "test_metrics.json").read_text())
    assert 0 <= report["all"]["miou"] <= 1 and (tmp_path / "run" / "S002_pred.npy").exists()
