import numpy as np
import pytest

from apairo_transform import ApplyPose
from apairo.core.sample import Sample


def pose_xyz(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """T_world_from_ego for a pure translation (no rotation)."""
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def frame(points, pose) -> Sample:
    return Sample(data={"lidar": np.asarray(points, dtype=np.float32), "pose": pose})


class TestApplyPose:
    def test_identity_pose_passthrough(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = ApplyPose()(frame(pts, np.eye(4))).data["lidar"]
        np.testing.assert_allclose(out, pts, atol=1e-6)

    def test_translation_lifts_to_world(self):
        # A point at local (4, 0, 0) seen from an ego at world x=1 sits at world x=5.
        out = ApplyPose()(frame([[4.0, 0.0, 0.0]], pose_xyz(1.0))).data["lidar"]
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]], atol=1e-6)

    def test_rotation_lifts_to_world(self):
        # Ego yaws 90° at the origin; a local (1, 0, 0) point lands at world (0, 1, 0).
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        out = ApplyPose()(frame([[1.0, 0.0, 0.0]], T)).data["lidar"]
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_preserves_extra_columns_and_dtype(self):
        out = ApplyPose()(frame([[1.0, 0.0, 0.0, 0.7]], pose_xyz(2.0))).data["lidar"]
        assert out.shape == (1, 4)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0], [3.0, 0.0, 0.0, 0.7], atol=1e-6)

    def test_output_channel_keeps_original(self):
        s = ApplyPose(output="lidar_world")(frame([[4.0, 0.0, 0.0]], pose_xyz(1.0)))
        np.testing.assert_allclose(s.data["lidar"], [[4.0, 0.0, 0.0]], atol=1e-6)
        np.testing.assert_allclose(s.data["lidar_world"], [[5.0, 0.0, 0.0]], atol=1e-6)

    def test_quaternion_pose_accepted(self):
        # [tx ty tz qx qy qz qw], identity rotation, +1 in x.
        p = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        s = Sample(data={"lidar": np.array([[4.0, 0, 0]], np.float32), "pose": p})
        out = ApplyPose()(s).data["lidar"]
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]], atol=1e-5)

    def test_missing_lidar_passthrough(self):
        s = Sample(data={"pose": np.eye(4)})
        assert ApplyPose()(s) is s  # nothing to lift, untouched

    def test_missing_pose_raises(self):
        with pytest.raises(ValueError, match="pose"):
            ApplyPose()(Sample(data={"lidar": np.zeros((3, 3), dtype=np.float32)}))
