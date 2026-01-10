from __future__ import annotations

import numpy as np

from lidar_seg.io import IGNORE


def confusion(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = target != IGNORE
    idx = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def scores(cm: np.ndarray, class_names: tuple[str, ...]) -> dict[str, float]:
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.divide(inter, union, out=np.full(len(inter), np.nan), where=union > 0)
    out = {"miou": float(np.nanmean(iou)), "oa": float(inter.sum() / max(cm.sum(), 1))}
    out.update({f"iou_{name}": float(v) for name, v in zip(class_names, iou, strict=True)})
    return out
