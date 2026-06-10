"""Generic array-level transforms — dtype casting and similar primitives.

Typical usage::

    from apairo_transform import CastTo
    import numpy as np

    ds.transform("voxelised",         CastTo(np.float32))
    ds.transform("voxelised_trav_gt", CastTo(np.int64))
"""

from __future__ import annotations

import numpy as np


class CastTo:
    """Cast a numpy array to the specified dtype.

    Wraps ``numpy.asarray(x, dtype=dtype)`` as a named, picklable callable
    suitable for use with ``dataset.transform()``.

    Args:
        dtype: Target numpy dtype (e.g. ``np.float32``, ``np.int64``,
               ``"uint8"``).

    Example::

        ds.transform("voxelised",         CastTo(np.float32))
        ds.transform("voxelised_trav_gt", CastTo(np.int64))
    """

    def __init__(self, dtype: np.dtype) -> None:
        self._dtype = np.dtype(dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self._dtype)

    def __repr__(self) -> str:
        return f"CastTo({self._dtype})"
