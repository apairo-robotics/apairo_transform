import math

import numpy as np
import pytest

from apairo_transform.pointcloud import (
    RandomRotation,
    RandomFlip,
    RandomScale,
    RandomTranslation,
    GaussianNoise,
    RandomPointDrop,
)


def make_pc(n: int = 200, channels: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, channels)).astype(np.float64)


# ── RandomRotation ────────────────────────────────────────────────────────────

class TestRandomRotation:
    def test_shape_preserved(self):
        pc = make_pc(100, channels=5)
        out = RandomRotation(axis="z", seed=0)(pc)
        assert out.shape == pc.shape

    def test_distances_preserved(self):
        pc = make_pc(100)
        out = RandomRotation(axis="z", seed=1)(pc)
        before = np.linalg.norm(pc[:, :3], axis=1)
        after = np.linalg.norm(out[:, :3], axis=1)
        np.testing.assert_allclose(before, after, atol=1e-10)

    def test_z_rotation_preserves_z(self):
        pc = make_pc(100)
        out = RandomRotation(axis="z", seed=2)(pc)
        np.testing.assert_allclose(pc[:, 2], out[:, 2], atol=1e-10)

    def test_x_rotation_preserves_x(self):
        pc = make_pc(100)
        out = RandomRotation(axis="x", seed=3)(pc)
        np.testing.assert_allclose(pc[:, 0], out[:, 0], atol=1e-10)

    def test_y_rotation_preserves_y(self):
        pc = make_pc(100)
        out = RandomRotation(axis="y", seed=4)(pc)
        np.testing.assert_allclose(pc[:, 1], out[:, 1], atol=1e-10)

    def test_non_xyz_channels_unchanged(self):
        pc = make_pc(100, channels=5)
        out = RandomRotation(axis="z", seed=5)(pc)
        np.testing.assert_array_equal(pc[:, 3:], out[:, 3:])

    def test_so3_preserves_distances(self):
        pc = make_pc(100)
        out = RandomRotation(axis="so3", seed=6)(pc)
        before = np.linalg.norm(pc[:, :3], axis=1)
        after = np.linalg.norm(out[:, :3], axis=1)
        np.testing.assert_allclose(before, after, atol=1e-10)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            RandomRotation(axis="w")

    def test_invalid_max_angle(self):
        with pytest.raises(ValueError):
            RandomRotation(max_angle=0.0)

    def test_no_original_mutation(self):
        pc = make_pc(50)
        original = pc.copy()
        RandomRotation(axis="z", seed=0)(pc)
        np.testing.assert_array_equal(pc, original)


# ── RandomFlip ────────────────────────────────────────────────────────────────

class TestRandomFlip:
    def test_always_flips_x(self):
        pc = make_pc(100)
        out = RandomFlip(axis="x", p=1.0, seed=0)(pc)
        np.testing.assert_allclose(out[:, 0], -pc[:, 0])
        np.testing.assert_allclose(out[:, 1:], pc[:, 1:])

    def test_always_flips_y(self):
        pc = make_pc(100)
        out = RandomFlip(axis="y", p=1.0, seed=0)(pc)
        np.testing.assert_allclose(out[:, 1], -pc[:, 1])
        np.testing.assert_allclose(out[:, 0], pc[:, 0])
        np.testing.assert_allclose(out[:, 2:], pc[:, 2:])

    def test_never_flips_with_p0(self):
        pc = make_pc(100)
        out = RandomFlip(axis="x", p=0.0, seed=0)(pc)
        np.testing.assert_array_equal(out, pc)

    def test_non_xyz_channels_unchanged(self):
        pc = make_pc(100, channels=5)
        out = RandomFlip(axis="x", p=1.0)(pc)
        np.testing.assert_array_equal(pc[:, 3:], out[:, 3:])

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            RandomFlip(axis="z")

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            RandomFlip(p=1.5)

    def test_no_original_mutation(self):
        pc = make_pc(50)
        original = pc.copy()
        RandomFlip(axis="x", p=1.0)(pc)
        np.testing.assert_array_equal(pc, original)


# ── RandomScale ───────────────────────────────────────────────────────────────

class TestRandomScale:
    def test_shape_preserved(self):
        pc = make_pc(100)
        out = RandomScale(range=(0.9, 1.1), seed=0)(pc)
        assert out.shape == pc.shape

    def test_xyz_scaled_within_range(self):
        pc = make_pc(100)
        out = RandomScale(range=(0.9, 1.1), seed=1)(pc)
        # For each point, ratio xyz_out / xyz_in should be constant and in [0.9, 1.1]
        # Pick a non-zero element
        i = 0
        ratio = out[i, 0] / pc[i, 0]
        assert 0.9 <= ratio <= 1.1
        # All xyz scaled by the same factor
        np.testing.assert_allclose(out[:, :3], pc[:, :3] * ratio, atol=1e-12)

    def test_extra_channels_unchanged(self):
        pc = make_pc(100, channels=5)
        out = RandomScale(range=(0.5, 2.0), seed=2)(pc)
        np.testing.assert_array_equal(out[:, 3:], pc[:, 3:])

    def test_invalid_range_lo_le_0(self):
        with pytest.raises(ValueError):
            RandomScale(range=(0.0, 1.0))

    def test_invalid_range_lo_ge_hi(self):
        with pytest.raises(ValueError):
            RandomScale(range=(1.5, 1.0))

    def test_no_original_mutation(self):
        pc = make_pc(50)
        original = pc.copy()
        RandomScale(range=(0.9, 1.1), seed=0)(pc)
        np.testing.assert_array_equal(pc, original)


# ── RandomTranslation ─────────────────────────────────────────────────────────

class TestRandomTranslation:
    def test_shape_preserved(self):
        pc = make_pc(100)
        out = RandomTranslation(sigma=0.5, seed=0)(pc)
        assert out.shape == pc.shape

    def test_all_points_shifted_by_same_vector(self):
        pc = make_pc(100)
        out = RandomTranslation(sigma=1.0, seed=3)(pc)
        diff = out[:, :3] - pc[:, :3]
        # all rows of diff must be the same (single shared shift)
        assert diff.std(axis=0).max() < 1e-10

    def test_extra_channels_unchanged(self):
        pc = make_pc(100, channels=5)
        out = RandomTranslation(sigma=0.5, seed=4)(pc)
        np.testing.assert_array_equal(out[:, 3:], pc[:, 3:])

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            RandomTranslation(sigma=0.0)

    def test_no_original_mutation(self):
        pc = make_pc(50)
        original = pc.copy()
        RandomTranslation(sigma=0.2, seed=0)(pc)
        np.testing.assert_array_equal(pc, original)


# ── GaussianNoise ─────────────────────────────────────────────────────────────

class TestGaussianNoise:
    def test_shape_preserved(self):
        pc = make_pc(500)
        out = GaussianNoise(sigma=0.01, seed=0)(pc)
        assert out.shape == pc.shape

    def test_noise_per_point(self):
        pc = make_pc(500)
        out = GaussianNoise(sigma=0.1, seed=1)(pc)
        diff = out[:, :3] - pc[:, :3]
        # Each point has a different shift — not all equal
        assert not np.allclose(diff, diff[0])

    def test_extra_channels_unchanged(self):
        pc = make_pc(100, channels=5)
        out = GaussianNoise(sigma=0.1, seed=2)(pc)
        np.testing.assert_array_equal(out[:, 3:], pc[:, 3:])

    def test_noise_magnitude_reasonable(self):
        pc = make_pc(10000)
        out = GaussianNoise(sigma=0.05, seed=3)(pc)
        diff = out[:, :3] - pc[:, :3]
        std = diff.std()
        assert abs(std - 0.05) < 0.01, f"Expected std ≈ 0.05, got {std:.4f}"

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            GaussianNoise(sigma=-0.1)

    def test_no_original_mutation(self):
        pc = make_pc(50)
        original = pc.copy()
        GaussianNoise(sigma=0.01, seed=0)(pc)
        np.testing.assert_array_equal(pc, original)


# ── RandomPointDrop ───────────────────────────────────────────────────────────

class TestRandomPointDrop:
    def test_p0_keeps_all(self):
        pc = make_pc(200)
        out = RandomPointDrop(p=0.0, seed=0)(pc)
        np.testing.assert_array_equal(out, pc)

    def test_drops_roughly_p_fraction(self):
        pc = make_pc(10000)
        out = RandomPointDrop(p=0.2, seed=42)(pc)
        drop_rate = 1 - len(out) / len(pc)
        assert abs(drop_rate - 0.2) < 0.02, f"Expected ≈20% drop, got {drop_rate:.2%}"

    def test_channels_preserved(self):
        pc = make_pc(500, channels=6)
        out = RandomPointDrop(p=0.1, seed=5)(pc)
        assert out.shape[1] == 6

    def test_invalid_p_ge_1(self):
        with pytest.raises(ValueError):
            RandomPointDrop(p=1.0)

    def test_invalid_p_negative(self):
        with pytest.raises(ValueError):
            RandomPointDrop(p=-0.1)

    def test_high_drop_rate(self):
        pc = make_pc(1000)
        out = RandomPointDrop(p=0.99, seed=0)(pc)
        assert len(out) < len(pc)
