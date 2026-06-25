"""Temporal point-cloud accumulation — a stateful, sample-level transform.

Densifies a sparse single-shot scan by stacking the last few frames into the
current ego frame, using each frame's pose to place it correctly::

    ds.transform(AccumulateFrames(lidar="lidar", pose="pose", num_frames=5))

Unlike the other point-cloud transforms (pure ``(N, D) -> (N, D)`` callables),
this one reads **two** channels — the scan and the pose — so it is registered as
a *sample-level* transform (``ds.transform(fn)``, not ``ds.transform(key, fn)``)
and keeps the frames it has seen in an internal rolling buffer.

.. warning::
    Accumulation is **stateful and order-dependent**: it assumes the dataset is
    consumed sequentially (one frame after the next, in index order). Shuffling
    the dataset, or reading it from a multi-worker ``DataLoader``, scrambles the
    buffer and yields meaningless results. Accumulate first (sequential pass),
    then shuffle/cache downstream, and call :meth:`reset` between epochs.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from apairo_transform.pose.matrix import PoseTo4x4

_to_4x4 = PoseTo4x4()


class AccumulateFrames:
    """Accumulate consecutive lidar scans into the current frame using poses.

    Maintains a rolling buffer of the most recently seen samples. On every
    call it appends the current ``(scan, pose)``, then gathers ``num_frames``
    frames spaced ``stride`` samples apart (the current frame plus the previous
    ``num_frames - 1``), transforms each past scan into the current ego frame and
    concatenates them into a single denser cloud.

    Poses are interpreted as ``T_world_from_ego`` — the transform mapping a point
    expressed in a frame's ego coordinates to the shared world frame (the usual
    odometry / SLAM convention). A past scan ``k`` is reprojected with
    ``T_cur_from_k = inv(T_world_from_cur) @ T_world_from_k`` and only its XYZ
    columns (first three) are moved; any extra columns (intensity, ...) ride
    along untouched.

    Early on — before enough frames have been seen — fewer frames are available
    and the result is simply shorter; it never errors or pads.

    Args:
        lidar:        Channel holding the ``(N, D)`` scan (``D >= 3``, XYZ first).
        pose:         Channel holding the ego pose. Any layout accepted by
                      :class:`~apairo_transform.pose.matrix.PoseTo4x4` works
                      (``(4, 4)``, ``(3, 4)``, ``(7,)`` quaternion, ``(6,)`` Euler).
        num_frames:   Total frames to stack, including the current one.
        stride:       Gap, in samples seen, between two accumulated frames
                      (``stride=1`` stacks every frame, ``stride=3`` every third).
        output:       Channel to write the accumulated cloud to. Defaults to
                      overwriting ``lidar``.
        time_channel: If true, append one extra column to every point holding
                      how many strides back its source frame is (``0`` for the
                      current frame, ``1`` for the previous kept frame, ...).
    """

    def __init__(
        self,
        lidar: str = "lidar",
        pose: str = "pose",
        num_frames: int = 5,
        stride: int = 1,
        output: Optional[str] = None,
        time_channel: bool = False,
    ) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._lidar = lidar
        self._pose = pose
        self._num_frames = num_frames
        self._stride = stride
        self._output = output if output is not None else lidar
        self._time_channel = time_channel
        # Enough history to reach num_frames frames spaced `stride` apart.
        self._buffer: deque = deque(maxlen=(num_frames - 1) * stride + 1)

    def reset(self) -> None:
        """Clear the frame buffer (call between epochs / sequences)."""
        self._buffer.clear()

    def __call__(self, sample):
        data = sample.data
        if self._lidar not in data:
            return sample
        if self._pose not in data:
            raise ValueError(
                f"AccumulateFrames needs pose channel {self._pose!r} to place "
                f"frames, but the sample only has {sorted(data)}."
            )

        scan = np.asarray(data[self._lidar])
        pose = _as_single_4x4(_to_4x4(data[self._pose]), self._pose)
        self._buffer.append((scan.copy(), pose))

        inv_cur = _invert_rigid(pose)  # T_cur_from_world

        clouds: list[np.ndarray] = []
        n = len(self._buffer)
        for step in range(self._num_frames):
            idx = n - 1 - step * self._stride
            if idx < 0:
                break
            past_scan, past_pose = self._buffer[idx]
            t_rel = inv_cur @ past_pose  # T_cur_from_past
            cloud = _apply_rigid(past_scan, t_rel)
            if self._time_channel:
                age = np.full((len(cloud), 1), step, dtype=cloud.dtype)
                cloud = np.concatenate([cloud, age], axis=1)
            clouds.append(cloud)

        data[self._output] = (
            clouds[0] if len(clouds) == 1 else np.concatenate(clouds, axis=0)
        )
        return sample

    def __repr__(self) -> str:
        return (
            f"AccumulateFrames(lidar={self._lidar!r}, pose={self._pose!r}, "
            f"num_frames={self._num_frames}, stride={self._stride}, "
            f"output={self._output!r}, time_channel={self._time_channel})"
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _as_single_4x4(pose: np.ndarray, key: str) -> np.ndarray:
    if pose.shape != (4, 4):
        raise ValueError(
            f"AccumulateFrames expects one 4x4 pose per sample in channel "
            f"{key!r}, got array of shape {pose.shape}."
        )
    return pose


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform — exact, no drift."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _apply_rigid(scan: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 ``T`` to the XYZ columns, preserving extra columns and dtype."""
    out = scan.astype(np.float64, copy=True)
    out[:, :3] = out[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out.astype(scan.dtype, copy=False)
