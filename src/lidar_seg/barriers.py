"""Road barrier extraction and measurement from a classified point cloud.

Points labelled as fence/guardrail are grouped into individual barriers (connected components
on a voxel grid). Each barrier is cut into 1 m stations along its main axis, and every station
gets the top height above the adjacent road surface. This is the measurement a guardrail
inspection needs: heights outside the allowed band are flagged.

    lidar-barriers --tile data/toronto3d_prepared/L002.npz --labels runs/.../L002_pred.npy --out barriers/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class BarrierConfig:
    barrier_class: int = 7  # fence (Toronto-3D)
    ground_classes: tuple[int, ...] = (0, 1)  # road, road marking
    voxel: float = 0.3  # grid for connected components [m]
    min_length: float = 4.0  # shorter clusters are ignored [m]
    station_spacing: float = 1.0  # [m]
    ground_radius: float = 3.0  # road points used for the local surface [m]
    top_percentile: float = 95.0  # robust "top of rail"
    allowed_height: tuple[float, float] = (0.6, 0.9)  # set from the applicable standard [m]


@dataclass
class Station:
    s: float  # distance along the barrier [m]
    x: float
    y: float
    top_z: float
    ground_z: float | None
    height: float | None
    points: int


@dataclass
class Barrier:
    barrier_id: int
    length: float
    stations: list[Station] = field(default_factory=list)

    def heights(self) -> np.ndarray:
        return np.array([s.height for s in self.stations if s.height is not None])


def cluster(points: np.ndarray, voxel: float) -> np.ndarray:
    """Connected components of occupied voxels (26-neighbourhood); returns a label per point."""
    keys = np.floor((points - points.min(0)) / voxel).astype(np.int64)
    grid = np.zeros(keys.max(0) + 1, dtype=bool)
    grid[tuple(keys.T)] = True
    labelled, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    return labelled[tuple(keys.T)]


def road_surface_at(points: np.ndarray, xy: np.ndarray) -> float:
    """Height of a least-squares plane through nearby road points, evaluated at ``xy``.
    A plane follows grade and crossfall, which a median of the neighbours does not."""
    design = np.c_[np.ones(len(points)), points[:, :2] - xy]
    coef, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    return float(coef[0])


def measure(xyz: np.ndarray, labels: np.ndarray, cfg: BarrierConfig) -> list[Barrier]:
    barrier_pts = xyz[labels == cfg.barrier_class]
    ground_pts = xyz[np.isin(labels, cfg.ground_classes)]
    if len(barrier_pts) == 0:
        return []
    ground_tree = cKDTree(ground_pts[:, :2]) if len(ground_pts) else None
    comp = cluster(barrier_pts, cfg.voxel)
    barriers = []
    for cid in np.unique(comp):
        pts = barrier_pts[comp == cid]
        center = pts[:, :2].mean(0)
        _, _, vt = np.linalg.svd(pts[:, :2] - center, full_matrices=False)
        axis = vt[0]
        s = (pts[:, :2] - center) @ axis
        length = float(s.max() - s.min())
        if length < cfg.min_length:
            continue
        barrier = Barrier(len(barriers), length)
        edges = np.arange(s.min(), s.max() + cfg.station_spacing, cfg.station_spacing)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = pts[(s >= lo) & (s < hi)]
            if len(sel) < 5:
                continue
            xy = sel[:, :2].mean(0)
            top = float(np.percentile(sel[:, 2], cfg.top_percentile))
            ground = None
            if ground_tree is not None:
                near = ground_tree.query_ball_point(xy, cfg.ground_radius)
                if len(near) >= 10:
                    ground = road_surface_at(ground_pts[near], xy)
            height = None if ground is None else top - ground
            barrier.stations.append(
                Station(float(lo - s.min()), float(xy[0]), float(xy[1]), top, ground, height, len(sel))
            )
        barriers.append(barrier)
    return barriers


def summarize(barriers: list[Barrier], cfg: BarrierConfig) -> list[dict]:
    lo, hi = cfg.allowed_height
    out = []
    for b in barriers:
        h = b.heights()
        out.append(
            {
                "barrier_id": b.barrier_id,
                "length_m": round(b.length, 2),
                "stations": len(b.stations),
                "height_median_m": round(float(np.median(h)), 3) if len(h) else None,
                "height_min_m": round(float(h.min()), 3) if len(h) else None,
                "height_max_m": round(float(h.max()), 3) if len(h) else None,
                "stations_out_of_range": int(((h < lo) | (h > hi)).sum()) if len(h) else 0,
            }
        )
    return out


def to_geojson(barriers: list[Barrier], cfg: BarrierConfig, offset: np.ndarray, crs: str) -> dict:
    """Barrier centre lines (LineString) with summary properties, in the source CRS."""
    features = []
    for b, summary in zip(barriers, summarize(barriers, cfg), strict=True):
        coords = [[s.x + offset[0], s.y + offset[1]] for s in b.stations]
        if len(coords) >= 2:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": summary,
                }
            )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tile", type=Path, required=True, help="prepared .npz tile")
    parser.add_argument(
        "--labels", type=Path, help=".npy predictions; default: the tile's ground-truth labels"
    )
    parser.add_argument("--out", type=Path, default=Path("barriers"))
    parser.add_argument("--crs", default="EPSG:32617", help="Toronto-3D is UTM zone 17N")
    args = parser.parse_args(argv)
    from lidar_seg.datasets.toronto3d import UTM_OFFSET

    archive = np.load(args.tile)
    labels = np.load(args.labels) if args.labels else archive["labels"]
    cfg = BarrierConfig()
    barriers = measure(archive["xyz"].astype(np.float64), labels, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "stations.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["barrier_id", "s_m", "x", "y", "top_z", "ground_z", "height_m", "points"])
        for b in barriers:
            for st in b.stations:
                writer.writerow(
                    [
                        b.barrier_id,
                        round(st.s, 2),
                        round(st.x + UTM_OFFSET[0], 3),
                        round(st.y + UTM_OFFSET[1], 3),
                        round(st.top_z, 3),
                        st.ground_z,
                        st.height,
                        st.points,
                    ]
                )
    geojson = to_geojson(barriers, cfg, UTM_OFFSET, args.crs)
    (args.out / "barriers.geojson").write_text(json.dumps(geojson))
    print(json.dumps(summarize(barriers, cfg), indent=2))


if __name__ == "__main__":
    main()
