"""Lift a single scan into the world frame using its per-frame pose.

The stateless, mono-frame counterpart of
:class:`~apairo_transform.pointcloud.accumulate.AccumulateFrames`: where
``AccumulateFrames`` reprojects a rolling *window* of past scans into the current
ego frame, ``ApplyPose`` takes the **one** scan of a sample and pushes its XYZ
out to the shared world frame using that sample's pose::

    ds.transform("odom", PoseTo4x4())          # if the pose isn't already 4x4
    ds.transform(ApplyPose(lidar="lidar", pose="odom"))

Like ``AccumulateFrames`` it reads **two** channels -- the scan and the pose --
so it is a *sample-level* transform (``ds.transform(fn)``, not
``ds.transform(key, fn)``). It holds no state, so unlike ``AccumulateFrames`` it
is safe under shuffling and multi-worker ``DataLoader``s.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from apairo import Sample

from apairo_transform.pose.matrix import (
    PoseTo4x4,
    apply_rigid as _apply_rigid,
    as_single_4x4 as _as_single_4x4,
)

_to_4x4 = PoseTo4x4()


class ApplyPose:
    """Lift a scan into the world frame with its ego pose (``world = pose @ xyz``).

    The pose is interpreted as ``T_world_from_ego`` -- the transform mapping a
    point from the ego frame to the shared world frame (the usual odometry /
    SLAM convention). Only the XYZ columns (first three) are moved; any extra
    columns (intensity, ...) ride along untouched, and the scan's dtype is
    preserved.

    .. note::
        The scan and the pose are assumed to live on the **same TF frame**: the
        pose's ego frame must be the frame the lidar XYZ are expressed in (i.e.
        ``pose`` is ``T_world_from_lidar`` when the points are in the lidar
        frame). If the cloud and the pose sit on different mounts, compose the
        static extrinsic first -- e.g. lift the points onto the pose's frame with
        ``ds.transform(lidar, TransformPoints(ds.calibration.get_tf(
        lidar_frame, pose_frame)))`` -- before applying this transform.

    Args:
        lidar:  Channel holding the ``(N, D)`` scan (``D >= 3``, XYZ first).
        pose:   Channel holding the ego pose. Any layout accepted by
                :class:`~apairo_transform.pose.matrix.PoseTo4x4` works
                (``(4, 4)``, ``(3, 4)``, ``(7,)`` quaternion, ``(6,)`` Euler).
        output: Channel to write the world-frame cloud to. Defaults to
                overwriting ``lidar``.
    """

    def __init__(
        self,
        lidar: str = "lidar",
        pose: str = "pose",
        output: Optional[str] = None,
    ) -> None:
        self._lidar = lidar
        self._pose = pose
        self._output = output if output is not None else lidar

    def __call__(self, sample: Sample) -> Sample:
        data = sample.data
        if self._lidar not in data:
            return sample
        if self._pose not in data:
            raise ValueError(
                f"ApplyPose needs pose channel {self._pose!r} to lift the cloud, "
                f"but the sample only has {sorted(data)}."
            )

        scan = np.asarray(data[self._lidar])
        pose = _as_single_4x4(_to_4x4(data[self._pose]), self._pose)
        data[self._output] = _apply_rigid(scan, pose)  # ego -> world
        return sample

    def __repr__(self) -> str:
        return (
            f"ApplyPose(lidar={self._lidar!r}, pose={self._pose!r}, "
            f"output={self._output!r})"
        )
