"""Label transforms — applied at access time to per-point or per-pixel label arrays.

Typical usage::

    mapping = {0: 255, 1: 0, 2: 0, 3: 1}   # remap class IDs
    ds = ds.transform("labels", RemapLabels(mapping))
    ds = ds.transform("labels", MaskLabels(keep={0, 1}, ignore_value=255))
"""

from __future__ import annotations

import numpy as np


def _holding(dtype: np.dtype, values) -> np.dtype:
    """*dtype*, widened if needed to hold every integer in *values*. A value
    that does not fit used to wrap (300 written as 44 in uint8) or raise,
    depending on the numpy version."""
    if dtype.kind not in "iu":
        return dtype
    for value in values:
        dtype = np.promote_types(dtype, np.min_scalar_type(int(value)))
    return dtype


class RemapLabels:
    """Remap integer label values according to a dictionary.

    Unknown labels (not present in ``mapping``) are set to ``default``.

    The result keeps the labels' integer type when it holds every target
    value and ``default``; otherwise it is the next type that does (uint8
    labels remapped to ids above 255 come out uint16).

    Args:
        mapping:       ``{old_id: new_id}`` mapping applied element-wise.
        default:       Value assigned to labels absent from ``mapping``.
                       Defaults to ``255`` (common ignore index).
    """

    def __init__(self, mapping: dict[int, int], default: int = 255) -> None:
        if not mapping:
            raise ValueError("mapping must not be empty")
        # Keys as integers: a recipe written to JSON hands them back as text.
        self._mapping = {int(k): int(v) for k, v in mapping.items()}
        self._default = int(default)

    def get_config(self) -> dict:
        """What rebuilds it (a saved chain stores this, keys as text)."""
        return {
            "mapping": {str(k): v for k, v in self._mapping.items()},
            "default": self._default,
        }

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        dtype = _holding(labels.dtype, [self._default, *self._mapping.values()])
        out = np.full(labels.shape, self._default, dtype=dtype)
        for src, dst in self._mapping.items():
            out[labels == src] = dst
        return out

    def __repr__(self) -> str:
        return f"RemapLabels(mapping={self._mapping}, default={self._default})"


class MaskLabels:
    """Set any label NOT in ``keep`` to ``ignore_value``.

    Useful for discarding rare or unknown classes before training.

    The result keeps the labels' integer type when it holds ``ignore_value``;
    otherwise it is the next type that does (int8 labels with the default 255
    come out int16).

    Args:
        keep:          Set of label IDs to preserve.
        ignore_value:  Value assigned to masked-out labels.  Defaults to ``255``.
    """

    def __init__(self, keep: set[int], ignore_value: int = 255) -> None:
        if not keep:
            raise ValueError("keep must not be empty")
        self._keep = set(keep)
        self._ignore = ignore_value

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        labels = labels.astype(_holding(labels.dtype, [self._ignore]), copy=True)
        mask = ~np.isin(labels, list(self._keep))
        labels[mask] = self._ignore
        return labels

    def __repr__(self) -> str:
        return f"MaskLabels(keep={sorted(self._keep)}, ignore_value={self._ignore})"
