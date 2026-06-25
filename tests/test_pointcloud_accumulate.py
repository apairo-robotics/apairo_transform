import numpy as np
import pytest

from apairo_transform import AccumulateFrames
from apairo.core.sample import Sample


def pose_xyz(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """T_world_from_ego for a pure translation (no rotation)."""
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def frame(points: np.ndarray, pose: np.ndarray) -> Sample:
    return Sample(data={"lidar": np.asarray(points, dtype=np.float32), "pose": pose})


class TestAccumulateFrames:
    def test_single_frame_passthrough(self):
        acc = AccumulateFrames(num_frames=3, stride=1)
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = acc(frame(pts, pose_xyz(0.0))).data["lidar"]
        np.testing.assert_allclose(out, pts, atol=1e-5)

    def test_translation_alignment(self):
        # A landmark sits at world x=5. The ego moves +1 in x each frame, so the
        # landmark's local x-coordinate shrinks: 5, then 4. After accumulation in
        # the *current* (second) frame both observations must land on x=4.
        acc = AccumulateFrames(num_frames=2, stride=1)
        acc(frame([[5.0, 0.0, 0.0]], pose_xyz(0.0)))
        out = acc(frame([[4.0, 0.0, 0.0]], pose_xyz(1.0))).data["lidar"]
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], atol=1e-5)

    def test_grows_then_caps(self):
        acc = AccumulateFrames(num_frames=3, stride=1)
        sizes = []
        for i in range(5):
            out = acc(frame([[float(i), 0.0, 0.0]], pose_xyz(float(i)))).data["lidar"]
            sizes.append(len(out))
        assert sizes == [1, 2, 3, 3, 3]  # builds up, then capped at num_frames

    def test_stride_skips_frames(self):
        acc = AccumulateFrames(num_frames=2, stride=2)
        for i in range(3):
            out = acc(frame([[float(i), 0.0, 0.0]], pose_xyz(float(i)))).data["lidar"]
        # current frame (i=2) + frame two back (i=0); the i=1 frame is skipped.
        assert len(out) == 2

    def test_preserves_extra_columns(self):
        acc = AccumulateFrames(num_frames=2, stride=1)
        acc(frame([[1.0, 0.0, 0.0, 0.7]], pose_xyz(0.0)))
        out = acc(frame([[0.0, 0.0, 0.0, 0.3]], pose_xyz(1.0))).data["lidar"]
        assert out.shape == (2, 4)
        # intensity column untouched; current frame first.
        np.testing.assert_allclose(out[:, 3], [0.3, 0.7], atol=1e-5)

    def test_time_channel(self):
        acc = AccumulateFrames(num_frames=3, stride=1, time_channel=True)
        acc(frame([[0.0, 0.0, 0.0]], pose_xyz(0.0)))
        acc(frame([[0.0, 0.0, 0.0]], pose_xyz(1.0)))
        out = acc(frame([[0.0, 0.0, 0.0]], pose_xyz(2.0))).data["lidar"]
        assert out.shape == (3, 4)
        np.testing.assert_array_equal(out[:, 3], [0, 1, 2])  # strides back

    def test_output_channel_keeps_original(self):
        acc = AccumulateFrames(num_frames=2, stride=1, output="lidar_acc")
        acc(frame([[5.0, 0.0, 0.0]], pose_xyz(0.0)))
        s = acc(frame([[4.0, 0.0, 0.0]], pose_xyz(1.0)))
        assert s.data["lidar"].shape == (1, 3)       # untouched
        assert s.data["lidar_acc"].shape == (2, 3)   # accumulated

    def test_rotation_alignment(self):
        # Ego yaws 90° between frames while staying at the origin. A point seen at
        # local (1, 0) in frame 0 is at world (1, 0); in the rotated frame 1 it
        # must reproject to local (0, -1).
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        T1 = np.eye(4)
        T1[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        acc = AccumulateFrames(num_frames=2, stride=1)
        acc(frame([[1.0, 0.0, 0.0]], np.eye(4)))
        out = acc(frame([[0.0, 0.0, 0.0]], T1)).data["lidar"]
        np.testing.assert_allclose(out[1], [0.0, -1.0, 0.0], atol=1e-6)

    def test_reset_clears_buffer(self):
        acc = AccumulateFrames(num_frames=3, stride=1)
        acc(frame([[0.0, 0.0, 0.0]], pose_xyz(0.0)))
        acc(frame([[0.0, 0.0, 0.0]], pose_xyz(1.0)))
        acc.reset()
        out = acc(frame([[0.0, 0.0, 0.0]], pose_xyz(2.0))).data["lidar"]
        assert len(out) == 1  # buffer empty → only current frame

    def test_missing_pose_raises(self):
        acc = AccumulateFrames()
        with pytest.raises(ValueError, match="pose"):
            acc(Sample(data={"lidar": np.zeros((3, 3), dtype=np.float32)}))

    def test_quaternion_pose_accepted(self):
        # identity quaternion [tx ty tz qx qy qz qw]
        acc = AccumulateFrames(num_frames=2, stride=1)
        p0 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        p1 = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        acc(Sample(data={"lidar": np.array([[5.0, 0, 0]], np.float32), "pose": p0}))
        s = Sample(data={"lidar": np.array([[4.0, 0, 0]], np.float32), "pose": p1})
        out = acc(s).data["lidar"]
        np.testing.assert_allclose(out, [[4.0, 0, 0], [4.0, 0, 0]], atol=1e-5)

    @pytest.mark.parametrize("bad", [{"num_frames": 0}, {"stride": 0}])
    def test_invalid_params(self, bad):
        with pytest.raises(ValueError):
            AccumulateFrames(**bad)
