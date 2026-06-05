"""Stochastic point-cloud augmentations for training data pipelines.

All augmentations operate on ``(N, D)`` float arrays and touch only the
first three columns (xyz).  Intensity and other channels are preserved.

Typical usage::

    from apairo.core import Compose
    from apairo_transform.pointcloud import RandomRotation, GaussianNoise

    aug = Compose([
        RandomRotation(axis="z"),
        RandomFlip(axis="x"),
        RandomScale(range=(0.95, 1.05)),
        GaussianNoise(sigma=0.01),
    ])
    ds.transform("lidar", aug)
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import numpy as np


class RandomRotation:
    """Rotate the point cloud around a fixed axis by a uniformly random angle.

    For outdoor LiDAR the standard choice is ``axis="z"`` (yaw only), which
    preserves the ground plane.  Use ``axis="so3"`` for full random SO3
    rotation (indoor / object-level tasks).

    Args:
        axis:      ``"x"``, ``"y"``, ``"z"`` or ``"so3"`` for a full random
                   rotation drawn uniformly from SO(3).
        max_angle: Half-range of the uniform distribution in radians.
                   Angle is sampled from ``[-max_angle, max_angle]``.
                   Ignored when ``axis="so3"``.
        seed:      Optional RNG seed.
    """

    def __init__(
        self,
        axis: Literal["x", "y", "z", "so3"] = "z",
        max_angle: float = math.pi,
        seed: Optional[int] = None,
    ) -> None:
        if axis not in ("x", "y", "z", "so3"):
            raise ValueError(f"axis must be 'x', 'y', 'z' or 'so3', got {axis!r}")
        if max_angle <= 0:
            raise ValueError(f"max_angle must be > 0, got {max_angle}")
        self._axis = axis
        self._max_angle = max_angle
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.array(pc, copy=True)
        R = self._sample_rotation()
        pc[:, :3] = pc[:, :3] @ R.T
        return pc

    def _sample_rotation(self) -> np.ndarray:
        if self._axis == "so3":
            return _random_so3(self._rng)
        angle = self._rng.uniform(-self._max_angle, self._max_angle)
        return _axis_angle_matrix(self._axis, angle)

    def __repr__(self) -> str:
        return f"RandomRotation(axis={self._axis!r}, max_angle={self._max_angle:.4f})"


class RandomFlip:
    """Mirror the point cloud along one axis with probability ``p``.

    Args:
        axis: ``"x"`` or ``"y"`` — the axis along which to negate coordinates.
        p:    Probability of actually flipping (default 0.5).
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        axis: Literal["x", "y"] = "x",
        p: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self._col = 0 if axis == "x" else 1
        self._axis = axis
        self._p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self._p:
            return pc
        pc = np.array(pc, copy=True)
        pc[:, self._col] *= -1
        return pc

    def __repr__(self) -> str:
        return f"RandomFlip(axis={self._axis!r}, p={self._p})"


class RandomScale:
    """Scale all xyz coordinates by a uniformly random factor.

    Args:
        range: ``(min_scale, max_scale)`` interval for the scale factor.
        seed:  Optional RNG seed.
    """

    def __init__(
        self,
        range: Tuple[float, float] = (0.95, 1.05),
        seed: Optional[int] = None,
    ) -> None:
        lo, hi = range
        if lo <= 0:
            raise ValueError(f"range[0] must be > 0, got {lo}")
        if lo >= hi:
            raise ValueError(f"range[0] must be < range[1], got {range}")
        self._lo = lo
        self._hi = hi
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.array(pc, copy=True)
        scale = self._rng.uniform(self._lo, self._hi)
        pc[:, :3] *= scale
        return pc

    def __repr__(self) -> str:
        return f"RandomScale(range=({self._lo}, {self._hi}))"


class RandomTranslation:
    """Translate all points by a random vector drawn from N(0, sigma²).

    Args:
        sigma: Standard deviation per axis (metres).
        seed:  Optional RNG seed.
    """

    def __init__(self, sigma: float = 0.2, seed: Optional[int] = None) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.array(pc, copy=True)
        shift = self._rng.normal(0.0, self._sigma, size=3)
        pc[:, :3] += shift
        return pc

    def __repr__(self) -> str:
        return f"RandomTranslation(sigma={self._sigma})"


class GaussianNoise:
    """Add independent Gaussian noise to xyz coordinates of every point.

    Simulates LiDAR range measurement noise.

    Args:
        sigma: Standard deviation of the noise (metres).
        seed:  Optional RNG seed.
    """

    def __init__(self, sigma: float = 0.01, seed: Optional[int] = None) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.array(pc, copy=True)
        pc[:, :3] += self._rng.normal(0.0, self._sigma, size=(len(pc), 3))
        return pc

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self._sigma})"


class RandomPointDrop:
    """Drop each point independently with probability ``p``.

    Simulates LiDAR dropouts caused by rain, dust, or absorption.

    Args:
        p:    Per-point drop probability.
        seed: Optional RNG seed.
    """

    def __init__(self, p: float = 0.05, seed: Optional[int] = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self._p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        keep = self._rng.random(len(pc)) >= self._p
        return pc[keep]

    def __repr__(self) -> str:
        return f"RandomPointDrop(p={self._p})"


# ── rotation helpers ──────────────────────────────────────────────────────────

def _axis_angle_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    # axis == "x"
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _random_so3(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation from SO(3) via quaternion sampling (Shoemake 1992)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
