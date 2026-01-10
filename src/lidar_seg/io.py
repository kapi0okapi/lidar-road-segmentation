"""Point cloud I/O (PLY, LAS/LAZ) and voxel-grid subsampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORE = 255  # label value for "no label" after subsampling / for unlabelled data


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N, 3) float64, metres
    rgb: np.ndarray | None = None  # (N, 3) uint8
    intensity: np.ndarray | None = None  # (N,) float32
    labels: np.ndarray | None = None  # (N,) uint8

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, idx: np.ndarray) -> PointCloud:
        pick = lambda a: None if a is None else a[idx]  # noqa: E731
        return PointCloud(self.xyz[idx], pick(self.rgb), pick(self.intensity), pick(self.labels))


def read_ply(path: str | Path, label_field: str | None = "scalar_Label") -> PointCloud:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    rgb = (
        np.stack([vertex[c] for c in ("red", "green", "blue")], axis=1).astype(np.uint8)
        if "red" in names
        else None
    )
    intensity_field = next((n for n in names if n.lower() in ("intensity", "scalar_intensity")), None)
    intensity = vertex[intensity_field].astype(np.float32) if intensity_field else None
    labels = vertex[label_field].astype(np.uint8) if label_field and label_field in names else None
    return PointCloud(xyz, rgb, intensity, labels)


def read_las(path: str | Path, use_classification: bool = False) -> PointCloud:
    import laspy

    las = laspy.read(str(path))
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    rgb = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        rgb16 = np.stack([las.red, las.green, las.blue], axis=1)
        rgb = (rgb16 >> 8 if rgb16.max() > 255 else rgb16).astype(np.uint8)
    intensity = np.asarray(las.intensity, dtype=np.float32)
    labels = np.asarray(las.classification, dtype=np.uint8) if use_classification else None
    return PointCloud(xyz, rgb, intensity, labels)


def read_points(path: str | Path) -> PointCloud:
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix in (".las", ".laz"):
        return read_las(path)
    raise ValueError(f"unsupported point cloud format: {suffix}")


def write_las(
    path: str | Path, cloud: PointCloud, classification: np.ndarray, extra: dict[str, np.ndarray] = ()
) -> None:
    """LAS/LAZ (by suffix) with predicted classes, readable in CloudCompare or QGIS."""
    import laspy

    header = laspy.LasHeader(point_format=3 if cloud.rgb is not None else 1, version="1.4")
    header.offsets = cloud.xyz.min(0)
    header.scales = np.array([0.001, 0.001, 0.001])
    for name, values in dict(extra).items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=values.dtype))
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    las.classification = classification.astype(np.uint8)
    if cloud.intensity is not None:
        las.intensity = np.clip(cloud.intensity, 0, 65535).astype(np.uint16)
    if cloud.rgb is not None:
        las.red, las.green, las.blue = (cloud.rgb.astype(np.uint16) << 8).T
    for name, values in dict(extra).items():
        las[name] = values
    las.write(str(path))


def grid_subsample(cloud: PointCloud, voxel: float, num_classes: int) -> tuple[PointCloud, np.ndarray]:
    """One point per occupied voxel: mean position / colour / intensity and majority label.
    Returns the subsampled cloud and, for every input point, the index of its voxel point
    (used to project predictions back to full resolution)."""
    keys = np.floor((cloud.xyz - cloud.xyz.min(0)) / voxel).astype(np.int64)
    dims = keys.max(0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[
        :, 2
    ]  # 1-D key: much faster than unique(axis=0)
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    m = len(counts)

    def mean(values: np.ndarray) -> np.ndarray:
        out = np.zeros((m, values.shape[1]))
        np.add.at(out, inverse, values)
        return out / counts[:, None]

    xyz = mean(cloud.xyz)
    rgb = mean(cloud.rgb.astype(np.float64)).round().astype(np.uint8) if cloud.rgb is not None else None
    intensity = (
        mean(cloud.intensity[:, None].astype(np.float64))[:, 0].astype(np.float32)
        if cloud.intensity is not None
        else None
    )
    labels = None
    if cloud.labels is not None:
        valid = cloud.labels < num_classes
        votes = np.bincount(inverse[valid] * num_classes + cloud.labels[valid], minlength=m * num_classes)
        votes = votes.reshape(m, num_classes)
        labels = np.where(votes.sum(1) > 0, votes.argmax(1), IGNORE).astype(np.uint8)
    return PointCloud(xyz, rgb, intensity, labels), inverse
