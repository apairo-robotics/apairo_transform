"""Fixes from the review of 0.3.0: randomness across workers, dtypes kept,
labels that hold their values, transforms a saved chain can declare."""
import os
import pickle
import sys
import types

import numpy as np
import pytest

import apairo_transform as at
from apairo_transform._random import Rng
from apairo_transform.label import MaskLabels, RemapLabels
from apairo_transform.pointcloud import (
    RandomRotation,
    RandomSubsample,
    RangeFilter,
    TransformPoints,
    VoxelDownsample,
)
from apairo_transform.pose import PoseTo4x4


def _yaw(transform):
    out = transform(np.array([[1.0, 0.0, 0.0]]))
    return float(np.degrees(np.arctan2(out[0, 1], out[0, 0])))


def test_a_seed_is_reproducible_in_one_process():
    a, b = RandomRotation(axis="z", seed=3), RandomRotation(axis="z", seed=3)
    assert [_yaw(a) for _ in range(4)] == [_yaw(b) for _ in range(4)]


def test_a_copy_in_another_process_draws_its_own_numbers(monkeypatch):
    """What a forked DataLoader worker holds: a copy of the transform, in a
    process with another pid. It must not replay the parent's draws."""
    parent = RandomRotation(axis="z")  # seed=None
    first = [_yaw(parent) for _ in range(4)]
    worker = pickle.loads(pickle.dumps(parent))
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    assert [_yaw(worker) for _ in range(4)] != first
    again = pickle.loads(pickle.dumps(parent))
    monkeypatch.setattr(os, "getpid", lambda: 4343)
    assert [_yaw(again) for _ in range(4)] != first


def test_workers_and_epochs_differ_with_a_seed_and_stay_reproducible(monkeypatch):
    def worker(seed):
        fake = types.ModuleType("torch")
        fake.utils = types.SimpleNamespace(
            data=types.SimpleNamespace(
                get_worker_info=lambda: types.SimpleNamespace(seed=seed)
            )
        )
        monkeypatch.setitem(sys.modules, "torch", fake)
        return [_yaw(RandomRotation(axis="z", seed=0)) for _ in range(3)]

    w0, w1 = worker(1000), worker(1001)  # two workers of one epoch
    assert w0 != w1
    assert worker(1000) == w0  # same worker seed (torch.manual_seed): same draws
    monkeypatch.delitem(sys.modules, "torch")


def test_range_filter_has_no_hidden_upper_bound():
    cloud = np.array([[2.0, 0, 0], [60.0, 0, 0], [50.0, 0, 0]])
    assert len(RangeFilter(min=1.0)(cloud)) == 3  # it used to cut at 50 m
    assert RangeFilter(min=55.0)(cloud)[:, 0].tolist() == [60.0]
    assert RangeFilter(max=50.0)(cloud)[:, 0].tolist() == [2.0]  # max is exclusive


def test_transform_points_keeps_its_own_matrix():
    matrix = np.eye(4)
    transform = TransformPoints(matrix)
    matrix[:3, 3] = 100.0
    np.testing.assert_array_equal(transform(np.zeros((1, 3))), np.zeros((1, 3)))


def test_labels_hold_the_values_asked_for():
    labels = np.array([0, 1, 7, 9], dtype=np.uint8)
    out = RemapLabels({0: 0, 1: 300})(labels)
    assert out.tolist() == [0, 300, 255, 255] and out.dtype == np.uint16
    assert RemapLabels({0: 0, 1: 1}, default=-1)(labels).tolist() == [0, 1, -1, -1]
    signed = labels.astype(np.int8)
    assert MaskLabels(keep={0, 1})(signed).tolist() == [0, 1, 255, 255]
    assert MaskLabels(keep={0, 1}, ignore_value=-100)(labels).tolist() == [0, 1, -100, -100]
    # Values that fit keep the dtype, and the input is not modified.
    kept = RemapLabels({0: 5, 1: 6})(labels)
    assert kept.dtype == np.uint8 and labels.tolist() == [0, 1, 7, 9]


def test_a_zero_quaternion_is_refused_by_name():
    poses = np.array([[0, 0, 0, 0, 0, 0, 1.0], [1, 2, 3, 0, 0, 0, 0.0]])
    with pytest.raises(ValueError, match="Pose 1 has a quaternion of norm 0"):
        PoseTo4x4()(poses)
    with pytest.raises(ValueError, match="quaternion of norm 0"):
        at.Se3Interp()(0.5, 0.0, poses[0], 1.0, poses[1])


def test_voxel_downsample_drops_points_without_a_position_and_can_be_seeded():
    cloud = np.array([[0.1, 0.1, 0.1, 1.0], [np.nan, 0.0, 0.0, 2.0], [1.1, 1.1, 1.1, 3.0]])
    out = VoxelDownsample(1.0)(cloud)
    assert out.shape == (2, 4) and np.isfinite(out).all()
    dense = np.random.default_rng(0).uniform(0, 3, size=(200, 3))
    a = VoxelDownsample(1.0, reduction="random", seed=5)(dense)
    b = VoxelDownsample(1.0, reduction="random", seed=5)(dense)
    np.testing.assert_array_equal(a, b)


def test_resize_keeps_the_image_dtype():
    pytest.importorskip("cv2")
    from apairo_transform.image import Resize

    floats = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    out = Resize(4, 4)(floats)
    assert out.dtype == np.float32 and 0.2 < out.mean() < 0.8  # it was all zeros
    depth = np.full((8, 8), 3000, dtype=np.uint16)
    assert Resize(4, 4, method="nearest")(depth).tolist() == [[3000] * 4] * 4


def test_transforms_kept_at_inference_can_be_saved(tmp_path):
    pytest.importorskip("apairo.core.chain")
    from apairo import load_chain
    from apairo.dataset.stream import StreamDataset

    def source():
        t = np.arange(3) * 0.1
        return StreamDataset(
            {"lidar": (t, [np.arange(40, dtype=np.float64).reshape(10, 4) + i for i in range(3)]),
             "labels": (t, [np.arange(10, dtype=np.uint8) for _ in range(3)])}
        )

    ds = source().synchronize(reference="lidar")
    ds = ds.transform("lidar", RandomSubsample(4, seed=1), in_place=False)
    ds = ds.transform("lidar", at.CastTo(np.float32), in_place=False)
    ds = ds.transform("labels", RemapLabels({0: 1, 1: 300}), in_place=False)
    path = ds.inference(["lidar", "labels"]).save(tmp_path / "chain.json")
    replay = load_chain(path).on(source())
    assert replay[0].data["lidar"].dtype == np.float32
    assert replay[0].data["lidar"].shape == (4, 4)
    assert replay[0].data["labels"].tolist()[:3] == [1, 300, 255]


def test_rng_forwards_generator_methods_and_declares_itself():
    rng = Rng(7)
    assert rng.get_config() == {"seed": 7}
    assert 0.0 <= rng.random() < 1.0 and len(rng.permutation(5)) == 5
    with pytest.raises(AttributeError):
        rng._missing
