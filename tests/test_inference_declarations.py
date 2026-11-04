"""What apairo_transform declares for ds.inference() (apairo core).

Augmentations are training-only (``inference = False``); the pose-aware
steps declare the channels they read and write, so the chain can be pruned
around them; AccumulateFrames keeps state and says so.
"""

import warnings

import numpy as np
import pytest

import apairo_transform as at

AUGMENTATIONS = [
    at.RandomRotation,
    at.RandomFlip,
    at.RandomScale,
    at.RandomTranslation,
    at.GaussianNoise,
    at.RandomPointDrop,
    at.ShufflePoints,
]


@pytest.mark.parametrize("cls", AUGMENTATIONS)
def test_augmentations_are_training_only(cls):
    assert cls.inference is False


@pytest.mark.parametrize("cls", [at.RangeFilter, at.VoxelDownsample, at.PoseTo4x4])
def test_deterministic_steps_are_kept(cls):
    assert getattr(cls, "inference", True) is True


def test_pose_steps_declare_their_channels():
    stack = at.StackByPose(lidar="velo", pose="odom", output="dense")
    assert (stack.input_keys, stack.output_keys) == (["velo", "odom"], ["dense"])
    apply = at.ApplyPose(lidar="velo", pose="odom")
    assert (apply.input_keys, apply.output_keys) == (["velo", "odom"], ["velo"])


def test_accumulate_frames_is_flagged_stateful():
    with pytest.warns(DeprecationWarning):
        assert at.AccumulateFrames().stateful is True


def test_a_lidar_chain_deploys_and_round_trips(tmp_path):
    StreamDataset = pytest.importorskip("apairo.dataset.stream").StreamDataset
    load_chain = pytest.importorskip("apairo.core.chain").load_chain
    from apairo.core import Compose

    def stream(offset=0.0):
        t = np.arange(5) * 0.1 + offset
        clouds = [np.random.default_rng(i).normal(size=(50, 3)) * 30 for i in range(5)]
        poses = [np.eye(4) for _ in range(5)]
        return StreamDataset({"lidar": (t, clouds), "pose": (t, poses)})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ds = stream().synchronize(reference="lidar")
        ds = ds.window(size=2, reduce=at.StackByPose())
        ds = ds.transform(
            "lidar",
            Compose([at.RangeFilter(max=40.0), at.RandomRotation(axis="z")]),
            in_place=False,
        )
        ds = ds.transform("lidar", at.RandomPointDrop(p=0.5), in_place=False)
        deployed = ds.inference("lidar")

    removed = {e.item for e in deployed.summary.removed}
    assert {"RandomPointDrop('lidar')", "RandomRotation"} <= removed
    assert not deployed.summary.flags
    chain = load_chain(deployed.save(tmp_path / "chain.json"))  # trusted by default
    a, b = deployed.on(stream(9.0)), chain.on(stream(9.0))
    for i in range(len(a)):
        np.testing.assert_array_equal(a[i].data["lidar"], b[i].data["lidar"])
