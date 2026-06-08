"""Point-cloud transforms — applied at access time to (N, D) float arrays.

Typical usage::

    ds.transform("lidar", RangeFilter(max=50.0))
    ds.transform("lidar", Compose([RangeFilter(max=50.0), RandomSubsample(4096)]))
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RangeFilter:
    """Keep only points whose distance from the origin is within [min, max].

    The distance metric is controlled by ``norm``, which accepts the same
    values as the ``ord`` argument of :func:`numpy.linalg.norm`:

    * ``2``       – L2 / Euclidean (default)
    * ``1``       – L1 / Manhattan
    * ``np.inf``  – L∞ / Chebyshev: ``max(|x|, |y|, |z|)``
    * any float p – Lp norm

    Args:
        min:  Minimum range (metres).  ``None`` disables the lower bound.
        max:  Maximum range (metres).  ``None`` disables the upper bound.
        norm: Norm order passed to ``numpy.linalg.norm``.  Defaults to ``2``.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = 50.0,
        norm: float = 2,
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
        ranges = np.linalg.norm(pc[:, :3], ord=self._norm, axis=1)
        mask = np.ones(len(pc), dtype=bool)
        if self._min is not None:
            mask &= ranges >= self._min
        if self._max is not None:
            mask &= ranges < self._max
        return mask

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        return np.asarray(pc)[self.compute_mask(pc)]

    def __repr__(self) -> str:
        return f"RangeFilter(min={self._min}, max={self._max}, norm={self._norm})"


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
