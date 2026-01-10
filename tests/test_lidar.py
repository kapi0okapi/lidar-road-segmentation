import numpy as np
import torch

from lidar_seg.barriers import BarrierConfig, measure, summarize
from lidar_seg.data import SphereDataset, load_tile, multilevel_indices, split_levels, tile_features
from lidar_seg.datasets.synthetic import road_scene, write_prepared
from lidar_seg.datasets.toronto3d import remap_labels
from lidar_seg.io import IGNORE, PointCloud, grid_subsample, read_las, write_las
from lidar_seg.metrics import confusion, scores
from lidar_seg.model import RandLANet


def test_grid_subsample_keeps_majority_label():
    xyz = np.array([[0.01, 0.01, 0], [0.02, 0.02, 0], [0.03, 0.01, 0], [1.0, 1.0, 1.0]])
    cloud = PointCloud(xyz, labels=np.array([2, 2, 5, IGNORE], np.uint8))
    sub, inverse = grid_subsample(cloud, voxel=0.1, num_classes=8)
    assert len(sub) == 2
    assert sub.labels[inverse[0]] == 2 and sub.labels[inverse[3]] == IGNORE
    np.testing.assert_allclose(sub.xyz[inverse[0]], [0.02, 0.0133333, 0], atol=1e-6)


def test_toronto_label_remap():
    np.testing.assert_array_equal(remap_labels(np.array([0, 1, 8, 9])), [IGNORE, 0, 7, IGNORE])


def test_nan_intensity_does_not_poison_features():
    intensity = np.array([1.0, 10.0, np.nan, 255.0], np.float32)
    features = tile_features({"xyz": np.zeros((4, 3)), "intensity": intensity}, ("intensity",))
    assert np.isfinite(features).all()


def test_multilevel_indices_shapes():
    xyz = np.random.default_rng(0).random((1024, 3)).astype(np.float32)
    levels = multilevel_indices(xyz, k=8, ratios=(4, 4, 2))
    assert [len(x) for x in levels["xyz"]] == [1024, 256, 64]
    assert levels["pool"][0].shape == (256, 8) and levels["up"][0].shape == (1024,)
    assert levels["up"][1].max() < 64  # level-1 points map onto the 64 points kept at level 2


def test_randla_forward_backward(tmp_path):
    root = write_prepared(tmp_path)
    tiles = [load_tile(p, ("rgb", "intensity")) for p in sorted(root.glob("*.npz"))]
    ds = SphereDataset(
        tiles, num_points=2048, k=8, ratios=(4, 4, 2), length=2, augment=True, class_balanced=True
    )
    batch = {k: torch.stack([ds[0][k], ds[1][k]]) for k in ds[0]}
    model = RandLANet(d_in=7, num_classes=8, d_out=(8, 16, 32))
    logits = model(batch["features"], split_levels(batch, 3))
    assert logits.shape == (2, 8, 2048)
    torch.nn.functional.cross_entropy(logits, batch["labels"], ignore_index=IGNORE).backward()


def test_barrier_height_on_synthetic_scene():
    xyz, _, _, labels = road_scene(rail_height=0.75, seed=1)
    barriers = summarize(measure(xyz, labels, BarrierConfig()), BarrierConfig())
    assert len(barriers) == 2
    for b in barriers:
        assert (
            b["length_m"] > 55 and abs(b["height_median_m"] - 0.75) < 0.05 and b["stations_out_of_range"] == 0
        )


def test_low_barrier_is_flagged():
    xyz, _, _, labels = road_scene(rail_height=0.5, seed=1)
    assert all(
        b["stations_out_of_range"] > 50
        for b in summarize(measure(xyz, labels, BarrierConfig()), BarrierConfig())
    )


def test_las_roundtrip(tmp_path):
    xyz, rgb, intensity, labels = road_scene(length=10, seed=0)
    cloud = PointCloud(xyz + [600000, 7000000, 0], rgb, intensity)
    write_las(tmp_path / "a.laz", cloud, labels)
    back = read_las(tmp_path / "a.laz", use_classification=True)
    assert len(back) == len(xyz) and np.abs(back.xyz - cloud.xyz).max() < 2e-3
    np.testing.assert_array_equal(back.labels, labels)


def test_scores():
    cm = confusion(np.array([0, 1, 1, 2]), np.array([0, 1, 2, IGNORE]), 3)
    s = scores(cm, ("a", "b", "c"))
    assert s["iou_a"] == 1.0 and s["iou_b"] == 0.5 and s["oa"] == 2 / 3
