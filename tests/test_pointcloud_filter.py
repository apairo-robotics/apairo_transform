import numpy as np
import pytest

from apairo_transform.pointcloud import RangeFilter, RandomSubsample, ShufflePoints, ChannelSelect


def make_pc(n: int = 100, channels: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, channels)).astype(np.float32)


# ── RangeFilter ───────────────────────────────────────────────────────────────

class TestRangeFilter:
    def test_max_only(self):
        pc = make_pc(200)
        pc[:, :3] /= np.linalg.norm(pc[:, :3], axis=1, keepdims=True)
        pc[:100, :3] *= 10.0   # inside  (range ≈ 10)
        pc[100:, :3] *= 100.0  # outside (range ≈ 100)
        out = RangeFilter(max=50.0)(pc)
        ranges = np.linalg.norm(out[:, :3], axis=1)
        assert np.all(ranges < 50.0)
        assert len(out) == 100

    def test_min_only(self):
        pc = make_pc(200)
        norms = np.linalg.norm(pc[:, :3], axis=1, keepdims=True)
        pc[:, :3] /= norms
        pc[:100, :3] *= 0.5   # too close (< 1 m)
        pc[100:, :3] *= 5.0   # far enough
        out = RangeFilter(min=1.0)(pc)
        ranges = np.linalg.norm(out[:, :3], axis=1)
        assert np.all(ranges >= 1.0)
        assert len(out) == 100

    def test_min_and_max(self):
        rng = np.random.default_rng(42)
        directions = rng.standard_normal((300, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        ranges = rng.uniform(0.1, 200.0, 300)
        pc = np.column_stack([directions * ranges[:, None], np.ones(300)])
        out = RangeFilter(min=5.0, max=50.0)(pc)
        out_ranges = np.linalg.norm(out[:, :3], axis=1)
        assert np.all(out_ranges >= 5.0)
        assert np.all(out_ranges < 50.0)

    def test_preserves_extra_channels(self):
        pc = make_pc(50, channels=6)
        pc[:, :3] /= np.linalg.norm(pc[:, :3], axis=1, keepdims=True)
        pc[:, :3] *= 10.0
        out = RangeFilter(max=20.0)(pc)
        assert out.shape[1] == 6

    def test_no_points_pass(self):
        pc = make_pc(50)
        pc[:, :3] /= np.linalg.norm(pc[:, :3], axis=1, keepdims=True)
        pc[:, :3] *= 200.0
        out = RangeFilter(max=10.0)(pc)
        assert len(out) == 0

    def test_invalid_min_negative(self):
        with pytest.raises(ValueError):
            RangeFilter(min=-1.0)

    def test_invalid_max_zero(self):
        with pytest.raises(ValueError):
            RangeFilter(max=0.0)

    def test_invalid_min_ge_max(self):
        with pytest.raises(ValueError):
            RangeFilter(min=10.0, max=5.0)


# ── RandomSubsample ───────────────────────────────────────────────────────────

class TestRandomSubsample:
    def test_downsamples_to_n(self):
        pc = make_pc(1000)
        out = RandomSubsample(n=256, seed=0)(pc)
        assert len(out) == 256
        assert out.shape[1] == pc.shape[1]

    def test_passthrough_when_smaller(self):
        pc = make_pc(100)
        out = RandomSubsample(n=500, seed=0)(pc)
        assert len(out) == 100
        np.testing.assert_array_equal(out, pc)

    def test_reproducible_with_seed(self):
        pc = make_pc(500)
        out1 = RandomSubsample(n=100, seed=42)(pc)
        out2 = RandomSubsample(n=100, seed=42)(pc)
        np.testing.assert_array_equal(out1, out2)

    def test_no_duplicates(self):
        pc = make_pc(500)
        out = RandomSubsample(n=250, seed=7)(pc)
        # all rows must be unique (rows from original)
        unique_rows = {tuple(r) for r in out}
        assert len(unique_rows) == 250

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            RandomSubsample(n=0)


# ── ShufflePoints ─────────────────────────────────────────────────────────────

class TestShufflePoints:
    def test_same_points_different_order(self):
        pc = make_pc(200, seed=1)
        out = ShufflePoints(seed=0)(pc)
        assert out.shape == pc.shape
        # same rows, possibly different order
        pc_sorted = np.sort(pc, axis=0)
        out_sorted = np.sort(out, axis=0)
        np.testing.assert_allclose(pc_sorted, out_sorted)

    def test_reproducible_with_seed(self):
        pc = make_pc(100)
        out1 = ShufflePoints(seed=99)(pc)
        out2 = ShufflePoints(seed=99)(pc)
        np.testing.assert_array_equal(out1, out2)

    def test_actually_shuffles(self):
        pc = make_pc(500, seed=5)
        out = ShufflePoints(seed=3)(pc)
        assert not np.array_equal(pc, out), "expected shuffle to change order"


# ── ChannelSelect ─────────────────────────────────────────────────────────────

class TestChannelSelect:
    def test_xyz_only(self):
        pc = make_pc(50, channels=6)
        out = ChannelSelect([0, 1, 2])(pc)
        assert out.shape == (50, 3)
        np.testing.assert_array_equal(out, pc[:, :3])

    def test_arbitrary_channels(self):
        pc = make_pc(50, channels=5)
        out = ChannelSelect([4, 0])(pc)
        assert out.shape == (50, 2)
        np.testing.assert_array_equal(out[:, 0], pc[:, 4])
        np.testing.assert_array_equal(out[:, 1], pc[:, 0])

    def test_single_channel(self):
        pc = make_pc(30, channels=4)
        out = ChannelSelect([2])(pc)
        assert out.shape == (30, 1)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            ChannelSelect([])
