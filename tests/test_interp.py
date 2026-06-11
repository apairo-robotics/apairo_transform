import numpy as np
import pytest

from apairo_transform.interp import LinearInterp, Se3Interp, _rot_to_quat, _slerp


# ── helpers ───────────────────────────────────────────────────────────────────

def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_z(angle: float) -> np.ndarray:
    """[qx, qy, qz, qw] for a rotation of *angle* around z."""
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])


def random_rigid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.standard_normal(3)
    return T


# ── LinearInterp ──────────────────────────────────────────────────────────────

class TestLinearInterp:
    def test_midpoint(self):
        v = LinearInterp()(0.5, 0.0, np.array([0.0, 10.0]), 1.0, np.array([2.0, 20.0]))
        np.testing.assert_allclose(v, [1.0, 15.0])

    def test_endpoints(self):
        interp = LinearInterp()
        v0, v1 = np.array([1.0]), np.array([5.0])
        np.testing.assert_allclose(interp(0.0, 0.0, v0, 1.0, v1), v0)
        np.testing.assert_allclose(interp(1.0, 0.0, v0, 1.0, v1), v1)

    def test_asymmetric_bracket(self):
        # t = 1.0 in [0, 4] -> alpha 0.25
        v = LinearInterp()(1.0, 0.0, np.array([0.0]), 4.0, np.array([8.0]))
        np.testing.assert_allclose(v, [2.0])


# ── Se3Interp — quaternion form (7,) ─────────────────────────────────────────

class TestSe3Interp7D:
    def test_midpoint_rotation_and_translation(self):
        p0 = np.concatenate([[0.0, 0.0, 0.0], quat_z(0.0)])
        p1 = np.concatenate([[2.0, 0.0, 0.0], quat_z(np.pi / 2)])
        p = Se3Interp()(0.5, 0.0, p0, 1.0, p1)
        np.testing.assert_allclose(p[:3], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(p[3:], quat_z(np.pi / 4), atol=1e-12)

    def test_shortest_path_with_negated_quaternion(self):
        """q and -q encode the same rotation: results must be identical."""
        p0 = np.concatenate([np.zeros(3), quat_z(0.0)])
        p1a = np.concatenate([np.zeros(3), quat_z(np.pi / 2)])
        p1b = np.concatenate([np.zeros(3), -quat_z(np.pi / 2)])
        ra = Se3Interp()(0.5, 0.0, p0, 1.0, p1a)
        rb = Se3Interp()(0.5, 0.0, p0, 1.0, p1b)
        np.testing.assert_allclose(np.abs(ra[3:]), np.abs(rb[3:]), atol=1e-12)

    def test_output_quaternion_is_unit(self):
        p0 = np.concatenate([np.zeros(3), quat_z(0.1)])
        p1 = np.concatenate([np.ones(3), quat_z(2.5)])
        p = Se3Interp()(0.3, 0.0, p0, 1.0, p1)
        assert np.linalg.norm(p[3:]) == pytest.approx(1.0)


# ── Se3Interp — homogeneous form (4, 4) ──────────────────────────────────────

class TestSe3Interp4x4:
    def test_midpoint_is_half_rotation(self):
        T0, T1 = np.eye(4), np.eye(4)
        T1[:3, :3] = rot_z(np.pi / 2)
        T1[:3, 3] = [4.0, 0.0, 0.0]
        T = Se3Interp()(0.5, 0.0, T0, 1.0, T1)
        np.testing.assert_allclose(T[:3, :3], rot_z(np.pi / 4), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], [2.0, 0.0, 0.0], atol=1e-12)

    def test_result_is_rigid(self):
        T = Se3Interp()(0.37, 0.0, random_rigid(1), 1.0, random_rigid(2))
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0])

    def test_roundtrip_rot_to_quat(self):
        for seed in range(5):
            R = random_rigid(seed)[:3, :3]
            q = _rot_to_quat(R)
            assert np.linalg.norm(q) == pytest.approx(1.0)
            # slerp at the endpoints reproduces the inputs
            np.testing.assert_allclose(_slerp(q, q, 0.5), q, atol=1e-12)


class TestSe3InterpErrors:
    def test_bad_shape(self):
        with pytest.raises(ValueError, match="expects shape"):
            Se3Interp()(0.5, 0.0, np.zeros(3), 1.0, np.zeros(3))

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="different shapes"):
            Se3Interp()(0.5, 0.0, np.zeros(7), 1.0, np.eye(4))


# ── integration with ds.synchronize ──────────────────────────────────────────

class TestSynchronizeIntegration:
    def _pose_dataset(self):
        """Async in-memory dataset: 'scan' at 2 Hz, 'pose' (7D) at 1 Hz."""
        from apairo.core import AbstractDataset, Sample

        pose_ts = np.array([0.0, 1.0])
        poses = [
            np.concatenate([[0.0, 0.0, 0.0], quat_z(0.0)]),
            np.concatenate([[2.0, 0.0, 0.0], quat_z(np.pi / 2)]),
        ]
        scan_ts = np.array([0.0, 0.5, 1.0])

        class DS(AbstractDataset):
            def __init__(self):
                self.keys = ["scan", "pose"]
                self.timestamps = {"scan": scan_ts, "pose": pose_ts}
                self.loaders = {
                    "scan": [np.full((4, 3), i, dtype=np.float32) for i in range(3)],
                    "pose": poses,
                }

            def __len__(self):
                return 5

            def _load(self, idx):
                return Sample(data={}, timestamp=0.0)

        return DS()

    def test_pose_interpolated_at_scan_times(self):
        ds = self._pose_dataset()
        view = ds.synchronize(reference="scan", method={"pose": Se3Interp()})
        assert len(view) == 3

        mid = view[1].data["pose"]  # t = 0.5 between the two poses
        np.testing.assert_allclose(mid[:3], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(mid[3:], quat_z(np.pi / 4), atol=1e-12)

        # exact ticks return the stored poses untouched
        np.testing.assert_allclose(view[0].data["pose"][3:], quat_z(0.0))
        np.testing.assert_allclose(view[2].data["pose"][:3], [2.0, 0.0, 0.0])
