import numpy as np
import pytest

from apairo_transform import StackByPose, AccumulateFrames
from apairo.core.sample import Sample


def pose_xyz(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """T_world_from_ego for a pure translation (no rotation)."""
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def frame(points, pose, t=None) -> Sample:
    return Sample(
        data={"lidar": np.asarray(points, dtype=np.float32), "pose": pose},
        timestamp=t,
    )


class TestStackByPose:
    def test_single_frame_passthrough(self):
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = StackByPose()([frame(pts, pose_xyz(0.0))]).data["lidar"]
        np.testing.assert_allclose(out, pts, atol=1e-5)

    def test_translation_alignment(self):
        # Landmark at world x=5; ego moves +1 in x. Local x: 5 then 4. Reduced into
        # the anchor (newest, second) frame, both observations land on x=4.
        window = [
            frame([[5.0, 0.0, 0.0]], pose_xyz(0.0)),  # oldest
            frame([[4.0, 0.0, 0.0]], pose_xyz(1.0)),  # anchor (newest)
        ]
        out = StackByPose()(window).data["lidar"]
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], atol=1e-5)

    def test_anchor_first_in_output(self):
        # Concatenation is newest -> oldest (anchor first), like AccumulateFrames.
        window = [
            frame([[1.0, 0.0, 0.0, 0.7]], pose_xyz(0.0)),  # oldest
            frame([[0.0, 0.0, 0.0, 0.3]], pose_xyz(1.0)),  # anchor
        ]
        out = StackByPose()(window).data["lidar"]
        assert out.shape == (2, 4)
        np.testing.assert_allclose(out[:, 3], [0.3, 0.7], atol=1e-5)  # anchor first

    def test_time_channel_counts_strides_back(self):
        window = [frame([[0.0, 0.0, 0.0]], pose_xyz(float(i))) for i in range(3)]
        out = StackByPose(time_channel=True)(window).data["lidar"]
        assert out.shape == (3, 4)
        np.testing.assert_array_equal(out[:, 3], [0, 1, 2])  # anchor=0, older grows

    def test_rotation_alignment(self):
        # Ego yaws 90° between frames at the origin. A point at local (1, 0) in the
        # old frame reprojects to local (0, -1) in the anchor frame.
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        T1 = np.eye(4)
        T1[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        window = [frame([[1.0, 0.0, 0.0]], np.eye(4)), frame([[0.0, 0.0, 0.0]], T1)]
        out = StackByPose()(window).data["lidar"]
        np.testing.assert_allclose(out[1], [0.0, -1.0, 0.0], atol=1e-6)

    def test_output_channel_keeps_original(self):
        window = [
            frame([[5.0, 0.0, 0.0]], pose_xyz(0.0)),
            frame([[4.0, 0.0, 0.0]], pose_xyz(1.0)),
        ]
        s = StackByPose(output="lidar_acc")(window)
        assert s.data["lidar"].shape == (1, 3)      # untouched (anchor's)
        assert s.data["lidar_acc"].shape == (2, 3)  # stacked

    def test_carries_anchor_metadata(self):
        window = [
            frame([[0.0, 0.0, 0.0]], pose_xyz(0.0), t=10.0),
            frame([[0.0, 0.0, 0.0]], pose_xyz(1.0), t=11.0),
        ]
        s = StackByPose()(window)
        assert s.timestamp == 11.0  # anchor's timestamp
        np.testing.assert_array_equal(s.data["pose"], pose_xyz(1.0))  # anchor's pose

    def test_quaternion_pose_accepted(self):
        p0 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        p1 = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        window = [
            Sample(data={"lidar": np.array([[5.0, 0, 0]], np.float32), "pose": p0}),
            Sample(data={"lidar": np.array([[4.0, 0, 0]], np.float32), "pose": p1}),
        ]
        out = StackByPose()(window).data["lidar"]
        np.testing.assert_allclose(out, [[4.0, 0, 0], [4.0, 0, 0]], atol=1e-5)

    def test_empty_window_raises(self):
        with pytest.raises(ValueError, match="empty"):
            StackByPose()([])

    def test_missing_pose_raises(self):
        with pytest.raises(ValueError, match="pose"):
            StackByPose()([Sample(data={"lidar": np.zeros((3, 3), dtype=np.float32)})])

    def test_matches_accumulate_on_sequential_pass(self):
        # StackByPose over a window must equal AccumulateFrames' result at the same
        # anchor on a sequential pass -- same geometry, different driver.
        rng = np.random.default_rng(0)
        raw = [
            (rng.normal(size=(8, 4)).astype(np.float32), pose_xyz(float(i), 0.5 * i))
            for i in range(5)
        ]
        # AccumulateFrames overwrites each sample's lidar, so feed it its own copies.
        acc = AccumulateFrames(num_frames=3, stride=1, time_channel=True)
        for pts, pose in raw:
            acc_out = acc(frame(pts, pose)).data["lidar"]  # last anchor wins

        window = [frame(pts, pose) for pts, pose in raw[2:5]]  # same 3-frame window
        stack_out = StackByPose(time_channel=True)(window).data["lidar"]
        np.testing.assert_allclose(stack_out, acc_out, atol=1e-5)
