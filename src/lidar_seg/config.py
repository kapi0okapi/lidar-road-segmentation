"""Experiment configuration. YAML files in ``configs/`` override these defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

from lidar_seg.datasets.toronto3d import CLASSES


@dataclass
class DataConfig:
    root: str = "${DATA}/toronto3d_prepared"  # .npz tiles from lidar_seg.datasets.toronto3d
    train_tiles: tuple[str, ...] = ("L001", "L003", "L004")
    val_tiles: tuple[str, ...] = ("L002",)
    classes: tuple[str, ...] = CLASSES
    features: tuple[str, ...] = ("rgb", "intensity")  # plus centred xyz, always
    num_points: int = 40960  # points per sample
    k: int = 16  # neighbours per point
    ratios: tuple[int, ...] = (4, 4, 4, 4, 2)  # random-sampling ratio per encoder level
    batch_size: int = 4
    num_workers: int = 8
    steps_per_epoch: int = 500
    val_samples: int = 400


@dataclass
class ModelConfig:
    d_out: tuple[int, ...] = (16, 64, 128, 256, 512)


@dataclass
class OptimConfig:
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    max_grad_norm: float = 5.0
    class_weight_power: float = 0.5  # weights = 1 / frequency^power (0.5 = inverse square root)


@dataclass
class TrainConfig:
    name: str = "randla_toronto3d"
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
