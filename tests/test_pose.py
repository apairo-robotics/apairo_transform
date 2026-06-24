import math

import numpy as np
import pytest

from apairo_transform.pose import PoseTo4x4, InvertPose, lookup_transform


# ── helpers ───────────────────────────────────────────────────────────────────

def identity_4x4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def random_rigid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # random rotation via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.standard_normal(3)
    return T


# ── PoseTo4x4 ─────────────────────────────────────────────────────────────────

class TestPoseTo4x4:
    def test_passthrough_single_4x4(self):
        T = random_rigid(0)
        out = PoseTo4x4()(T)
        np.testing.assert_array_equal(out, T)

    def test_passthrough_batch_4x4(self):
        batch = np.stack([random_rigid(i) for i in range(5)])
        out = PoseTo4x4()(batch)
        np.testing.assert_array_equal(out, batch)

    def test_identity_quat_single(self):
        # identity quaternion [tx ty tz qx qy qz qw] = [0 0 0 0 0 0 1]
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        out = PoseTo4x4()(pose)
        np.testing.assert_allclose(out, identity_4x4(), atol=1e-12)

    def test_translation_quat(self):
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        out = PoseTo4x4()(pose)
        np.testing.assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(out[:3, :3], np.eye(3), atol=1e-12)
        assert out[3, 3] == pytest.approx(1.0)

    def test_batch_quat(self):
        poses = np.zeros((4, 7))
        poses[:, 6] = 1.0  # identity quaternion
        poses[:, :3] = np.arange(4 * 3).reshape(4, 3)
        out = PoseTo4x4()(poses)
        assert out.shape == (4, 4, 4)
        for i in range(4):
            np.testing.assert_allclose(out[i, :3, :3], np.eye(3), atol=1e-12)
            np.testing.assert_allclose(out[i, :3, 3], poses[i, :3], atol=1e-12)

    def test_quat_rotation_orthogonal(self):
        rng = np.random.default_rng(7)
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        pose = np.array([0.0, 0.0, 0.0, q[0], q[1], q[2], q[3]])
        out = PoseTo4x4()(pose)
        R = out[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_identity_euler_single(self):
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = PoseTo4x4()(pose)
        np.testing.assert_allclose(out, identity_4x4(), atol=1e-12)

    def test_translation_euler(self):
        pose = np.array([5.0, -3.0, 1.0, 0.0, 0.0, 0.0])
        out = PoseTo4x4()(pose)
        np.testing.assert_allclose(out[:3, 3], [5.0, -3.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(out[:3, :3], np.eye(3), atol=1e-12)

    def test_euler_rotation_orthogonal(self):
        pose = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.1])
        out = PoseTo4x4()(pose)
        R = out[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_batch_euler(self):
        poses = np.zeros((3, 6))
        out = PoseTo4x4()(poses)
        assert out.shape == (3, 4, 4)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            PoseTo4x4()(np.zeros((5,)))

    def test_last_row(self):
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        out = PoseTo4x4()(pose)
        np.testing.assert_array_equal(out[3], [0, 0, 0, 1])


# ── InvertPose ────────────────────────────────────────────────────────────────

class TestInvertPose:
    def test_identity_inverts_to_identity(self):
        T = identity_4x4()
        out = InvertPose()(T)
        np.testing.assert_allclose(out, identity_4x4(), atol=1e-12)

    def test_t_times_inv_is_identity(self):
        T = random_rigid(1)
        T_inv = InvertPose()(T)
        np.testing.assert_allclose(T @ T_inv, identity_4x4(), atol=1e-10)

    def test_inv_times_t_is_identity(self):
        T = random_rigid(2)
        T_inv = InvertPose()(T)
        np.testing.assert_allclose(T_inv @ T, identity_4x4(), atol=1e-10)

    def test_batch_inversion(self):
        batch = np.stack([random_rigid(i) for i in range(8)])
        inv = InvertPose()(batch)
        assert inv.shape == (8, 4, 4)
        for i in range(8):
            np.testing.assert_allclose(batch[i] @ inv[i], identity_4x4(), atol=1e-10)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            InvertPose()(np.eye(3))

    def test_invalid_shape_2d(self):
        with pytest.raises(ValueError):
            InvertPose()(np.zeros((5, 7)))

    def test_round_trip_quat(self):
        pose = np.array([1.0, -2.0, 3.0, 0.1, 0.2, 0.3, 0.9])
        T = PoseTo4x4()(pose)
        T_inv = InvertPose()(T)
        np.testing.assert_allclose(T @ T_inv, identity_4x4(), atol=1e-10)


# ── lookup_transform (deprecated; resolution now lives in apairo's Calibration) ─

class TestLookupTransform:
    def test_delegates_to_calibration_get_tf(self):
        from apairo.core.config import Calibration

        cal = {"a_to_b": random_rigid(1), "b_to_c": random_rigid(2)}
        with pytest.warns(DeprecationWarning):
            out = lookup_transform(cal, target="a", source="c")
        # same result as the canonical get_tf -- note the flipped argument order.
        np.testing.assert_allclose(out, Calibration(cal).get_tf("c", "a"), atol=1e-12)

    def test_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="get_tf"):
            lookup_transform({"a_to_b": random_rigid(0)}, target="a", source="a")
