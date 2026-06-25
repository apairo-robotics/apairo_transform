"""Point-cloud transforms — applied at access time to (N, D) float arrays.

Typical usage::

    ds.transform("lidar", RangeFilter(max=50.0))
    ds.transform("lidar", Compose([RangeFilter(max=50.0), RandomSubsample(4096)]))
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np

# Default norm: L2 / Euclidean distance on the XYZ columns.
_l2_norm: Callable[[np.ndarray], np.ndarray] = functools.partial(
    np.linalg.norm, axis=1
)


class RangeFilter:
    """Keep only points whose distance from the origin is within [min, max].

    The distance metric is controlled by ``norm``, a callable
    ``(ndarray of shape (N, 3)) -> (N,)`` that returns one distance per point.
    Defaults to the L2 / Euclidean norm.

    Common examples::

        RangeFilter(max=50.0)                                          # L2
        RangeFilter(max=50.0, norm=lambda pc: np.max(np.abs(pc), axis=1))   # L-inf
        RangeFilter(max=50.0, norm=lambda pc: np.sum(np.abs(pc), axis=1))   # L1

    Args:
        min:  Minimum range (metres).  ``None`` disables the lower bound.
        max:  Maximum range (metres).  ``None`` disables the upper bound.
        norm: Callable ``(N, 3) -> (N,)`` computing per-point distances.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = 50.0,
        norm: Callable[[np.ndarray], np.ndarray] = _l2_norm,
    ) -> None:
        if min is not None and min < 0:
            raise ValueError(f"min must be >= 0, got {min}")
        if max is not None and max <= 0:
            raise ValueError(f"max must be > 0, got {max}")
        if min is not None and max is not None and min >= max:
            raise ValueError(f"min ({min}) must be < max ({max})")
        self._min = min
        self._max = max
        self._norm = norm

    def compute_mask(self, pc: np.ndarray) -> np.ndarray:
        """Return the boolean keep-mask without applying it.

        Useful when multiple aligned arrays (e.g. point cloud + labels) must
        be filtered with the same mask::

            mask   = RangeFilter(max=50.0).compute_mask(pc)
            pc     = pc[mask]
            labels = labels[mask]
        """
        pc = np.asarray(pc)
        ranges = self._norm(pc[:, :3])
        mask = np.ones(len(pc), dtype=bool)
        if self._min is not None:
            mask &= ranges >= self._min
        if self._max is not None:
            mask &= ranges < self._max
        return mask

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        return np.asarray(pc)[self.compute_mask(pc)]

    def __repr__(self) -> str:
        return f"RangeFilter(min={self._min}, max={self._max}, norm={self._norm!r})"


class RandomSubsample:
    """Randomly subsample a point cloud to at most ``n`` points.

    If the cloud already has fewer than ``n`` points it is returned unchanged.

    Args:
        n:    Target number of points.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self._n = n
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        if len(pc) <= self._n:
            return pc
        idx = self._rng.choice(len(pc), self._n, replace=False)
        return pc[idx]

    def __repr__(self) -> str:
        return f"RandomSubsample(n={self._n})"


class ShufflePoints:
    """Randomly permute the order of points.

    Useful to break spatial ordering biases before feeding into models that
    process points sequentially.

    Args:
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        perm = self._rng.permutation(len(pc))
        return pc[perm]

    def __repr__(self) -> str:
        return "ShufflePoints()"


class HeightFilter:
    """Keep only points whose Z coordinate is within [min, max].

    Useful as a coarse ground-removal or sky-clipping step before registration
    or visualisation.

    Args:
        min: Minimum Z value (metres).  ``None`` disables the lower bound.
        max: Maximum Z value (metres).  ``None`` disables the upper bound.
    """

    def __init__(self, min: Optional[float] = None, max: Optional[float] = None) -> None:
        self._min = min
        self._max = max

    def compute_mask(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        mask = np.ones(len(pc), dtype=bool)
        if self._min is not None:
            mask &= pc[:, 2] >= self._min
        if self._max is not None:
            mask &= pc[:, 2] < self._max
        return mask

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        return np.asarray(pc)[self.compute_mask(pc)]

    def __repr__(self) -> str:
        return f"HeightFilter(min={self._min}, max={self._max})"


class TransformPoints:
    """Apply a fixed 4×4 rigid-body transform to the XYZ columns of a point cloud.

    Useful for static calibration transforms (e.g. lidar-to-camera extrinsic).
    For per-frame dynamic poses use a ``sample_transform`` that reads the pose
    key from the sample directly.

    Args:
        T: A ``(4, 4)`` homogeneous transformation matrix (float64).
    """

    def __init__(self, T: np.ndarray) -> None:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be (4, 4), got {T.shape}")
        self._R = T[:3, :3]
        self._t = T[:3, 3]

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc, dtype=np.float64).copy()
        pc[:, :3] = pc[:, :3] @ self._R.T + self._t
        return pc

    def __repr__(self) -> str:
        return "TransformPoints()"


class ChannelSelect:
    """Select a subset of columns from a point cloud array.

    Args:
        channels: Column indices to keep (e.g. ``[0, 1, 2]`` for xyz only).
    """

    def __init__(self, channels: list[int]) -> None:
        if not channels:
            raise ValueError("channels must not be empty")
        self._channels = list(channels)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        return pc[:, self._channels]

    def __repr__(self) -> str:
        return f"ChannelSelect(channels={self._channels})"
