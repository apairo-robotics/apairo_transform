"""In-memory voxel-grid downsampling — applied at access time to (N, D) arrays.

Unlike :mod:`apairo_preprocess`'s ``VoxelisePointCloud`` (which writes to disk),
this transform runs entirely in memory and is designed for use with
``dataset.transform()``.

Typical usage::

    from apairo_transform.pointcloud import VoxelDownsample

    ds.transform("lidar", VoxelDownsample(voxel_size=0.1, max_range=50.0))
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

_Reduction = Literal["centroid", "first", "random"]


class VoxelDownsample:
    """Voxel-grid downsampling of a point cloud.

    Partitions the point cloud into a regular grid of cubic voxels and retains
    exactly one representative point per occupied voxel.  All input channels
    (xyz + intensity / timestamp / …) are preserved in the output.

    Args:
        voxel_size: Edge length of each cubic voxel cell (metres).
        max_range:  Optional range filter applied before voxelisation.
                    Points farther than ``max_range`` metres from the origin
                    are discarded.  ``None`` keeps all points.
        reduction:  How to pick the representative point per voxel.

                    * ``"centroid"`` — mean of all points in the voxel (all
                      channels).
                    * ``"first"``    — first point in input order.
                    * ``"random"``   — one uniformly random point per voxel.
    """

    def __init__(
        self,
        voxel_size: float,
        max_range: Optional[float] = None,
        reduction: _Reduction = "centroid",
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_range is not None and max_range <= 0:
            raise ValueError(f"max_range must be positive, got {max_range}")
        if reduction not in ("centroid", "first", "random"):
            raise ValueError(
                f"reduction must be 'centroid', 'first', or 'random', got {reduction!r}"
            )
        self._voxel_size = float(voxel_size)
        self._max_range  = max_range
        self._reduction  = reduction

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc, dtype=np.float64)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        if self._max_range is not None:
            keep = np.linalg.norm(pc[:, :3], axis=1) < self._max_range
            pc = pc[keep]

        if len(pc) == 0:
            return pc

        coords   = np.floor(pc[:, :3] / self._voxel_size).astype(np.int32)
        _, inverse = np.unique(coords, axis=0, return_inverse=True)
        n_voxels = int(inverse.max()) + 1

        if self._reduction == "centroid":
            out    = np.zeros((n_voxels, pc.shape[1]), dtype=np.float64)
            counts = np.zeros(n_voxels, dtype=np.int64)
            np.add.at(out, inverse, pc)
            np.add.at(counts, inverse, 1)
            out /= counts[:, None]
            return out

        if self._reduction == "first":
            out = np.empty((n_voxels, pc.shape[1]), dtype=np.float64)
            out[inverse] = pc          # later writes win; reverse so first wins
            out_first = np.empty_like(out)
            for i in range(len(pc) - 1, -1, -1):
                out_first[inverse[i]] = pc[i]
            return out_first

        # random
        rng    = np.random.default_rng()
        chosen = np.empty(n_voxels, dtype=np.int64)
        perm   = rng.permutation(len(pc))
        chosen[inverse[perm]] = perm
        return pc[chosen]

    def __repr__(self) -> str:
        return (
            f"VoxelDownsample(voxel_size={self._voxel_size}, "
            f"max_range={self._max_range}, reduction={self._reduction!r})"
        )
