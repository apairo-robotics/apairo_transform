"""Pose-aware window reducer — temporal point-cloud accumulation.

``StackByPose`` densifies a scan with its predecessors. It receives a whole
window already and holds no state, so it works under random access,
shuffling and several ``DataLoader`` workers. It replaces the deprecated
:class:`~apairo_transform.pointcloud.accumulate.AccumulateFrames`, which kept a
rolling buffer and only worked on a sequential pass::

    from apairo_transform import StackByPose

    ds = ds.window(size=5, stride=1, reduce=StackByPose(time_channel=True))

It is a ``reduce`` callable for
:meth:`~apairo.core.abstract_dataset.AbstractDataset.window`: it takes the
window's samples (ordered oldest -> newest, the **last** being the anchor) and
returns a single sample whose scan is every frame reprojected into the anchor's
ego frame and concatenated into one denser cloud.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from apairo import Sample

from apairo_transform.pose.matrix import (
    PoseTo4x4,
    apply_rigid,
    as_single_4x4,
    invert_rigid,
)

_to_4x4 = PoseTo4x4()


class StackByPose:
    """Reduce a window of scans into the anchor frame using per-frame poses.

    Poses are interpreted as ``T_world_from_ego`` (the usual odometry / SLAM
    convention). The anchor is the newest frame (``samples[-1]``); each older
    scan ``k`` is reprojected with ``T_anchor_from_k = inv(T_world_from_anchor)
    @ T_world_from_k`` -- only its XYZ columns (first three) are moved, any extra
    columns (intensity, ...) ride along untouched -- and all scans are
    concatenated into a single cloud.

    The returned sample carries the anchor's other channels and timestamp; only
    the ``output`` channel (default: ``lidar``) is rewritten.

    Args:
        lidar:        Channel holding the ``(N, D)`` scan (``D >= 3``, XYZ first).
        pose:         Channel holding the ego pose. Any layout accepted by
                      :class:`~apairo_transform.pose.matrix.PoseTo4x4` works.
        output:       Channel to write the stacked cloud to. Defaults to
                      overwriting ``lidar``.
        time_channel: If true, append one column to every point holding how many
                      strides back its source frame is (``0`` for the anchor,
                      ``1`` for the previous kept frame, ...).
    """

    @property
    def input_keys(self) -> list[str]:
        """Channels read -- lets ds.inference() (apairo) prune around it."""
        return [self._lidar, self._pose]

    @property
    def output_keys(self) -> list[str]:
        """Channel written."""
        return [self._output]

    def __init__(
        self,
        lidar: str = "lidar",
        pose: str = "pose",
        output: Optional[str] = None,
        time_channel: bool = False,
    ) -> None:
        self._lidar = lidar
        self._pose = pose
        self._output = output if output is not None else lidar
        self._time_channel = time_channel

    def __call__(self, samples: list[Sample]) -> Sample:
        if not samples:
            raise ValueError("StackByPose received an empty window.")

        anchor = samples[-1]
        if self._lidar not in anchor.data:
            raise ValueError(
                f"StackByPose needs lidar channel {self._lidar!r}, but the anchor "
                f"sample only has {sorted(anchor.data)}."
            )
        if self._pose not in anchor.data:
            raise ValueError(
                f"StackByPose needs pose channel {self._pose!r} to place frames, "
                f"but the anchor sample only has {sorted(anchor.data)}."
            )

        anchor_pose = as_single_4x4(_to_4x4(anchor.data[self._pose]), self._pose)
        inv_anchor = invert_rigid(anchor_pose)  # T_anchor_from_world

        clouds: list[np.ndarray] = []
        n = len(samples)
        for k, sample in enumerate(reversed(samples)):  # k = strides back from anchor
            scan = np.asarray(sample.data[self._lidar])
            pose = as_single_4x4(_to_4x4(sample.data[self._pose]), self._pose)
            cloud = apply_rigid(scan, inv_anchor @ pose)  # into anchor ego frame
            if self._time_channel:
                age = np.full((len(cloud), 1), k, dtype=cloud.dtype)
                cloud = np.concatenate([cloud, age], axis=1)
            clouds.append(cloud)

        stacked = clouds[0] if n == 1 else np.concatenate(clouds, axis=0)

        data = dict(anchor.data)
        data[self._output] = stacked
        return Sample(data=data, timestamp=anchor.timestamp)

    def __repr__(self) -> str:
        return (
            f"StackByPose(lidar={self._lidar!r}, pose={self._pose!r}, "
            f"output={self._output!r}, time_channel={self._time_channel})"
        )
