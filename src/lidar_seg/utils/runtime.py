"""Training runtime helpers: seeding, device/precision selection, Slurm-aware stopping,
checkpoints and metric logging."""

from __future__ import annotations

import contextlib
import csv
import logging
import os
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds libraries that use the global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Precision:
    """Mixed precision policy: bf16 where supported (A100/H100), fp16 + loss scaling on
    older GPUs (V100), full precision elsewhere."""

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.dtype: torch.dtype | None = None
        if enabled and device.type == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)

    def autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        max_grad_norm: float | None = None,
    ) -> None:
        self.scaler.scale(loss).backward()
        if max_grad_norm:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)


class StopSignal:
    """Set when Slurm warns about the time limit (USR1, see ``--signal`` in the job script)
    or the job is cancelled (TERM). The training loop checks it and checkpoints."""

    def __init__(self) -> None:
        self.received: int | None = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        log.warning("received signal %s, stopping after the current step", signum)
        self.received = signum

    def __bool__(self) -> bool:
        return self.received is not None


def save_checkpoint(path: Path, **state: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a job killed mid-write never leaves a broken file


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


class MetricLogger:
    """Appends epoch metrics to ``metrics.csv`` and TensorBoard in the run directory."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "metrics.csv"
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(run_dir / "tb")
        except ImportError:
            log.info("tensorboard not installed, logging to CSV only")

    def log(self, step: int, metrics: dict[str, float]) -> None:
        row = {"epoch": step, **{k: round(float(v), 6) for k, v in metrics.items()}}
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
