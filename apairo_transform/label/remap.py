"""Label transforms — applied at access time to per-point or per-pixel label arrays.

Typical usage::

    mapping = {0: 255, 1: 0, 2: 0, 3: 1}   # remap class IDs
    ds.transform("labels", RemapLabels(mapping))
    ds.transform("labels", MaskLabels(keep={0, 1}, ignore_value=255))
"""

from __future__ import annotations

import numpy as np


class RemapLabels:
    """Remap integer label values according to a dictionary.

    Unknown labels (not present in ``mapping``) are set to ``default``.

    Args:
        mapping:       ``{old_id: new_id}`` mapping applied element-wise.
        default:       Value assigned to labels absent from ``mapping``.
                       Defaults to ``255`` (common ignore index).
    """

    def __init__(self, mapping: dict[int, int], default: int = 255) -> None:
        if not mapping:
            raise ValueError("mapping must not be empty")
        self._mapping = mapping
        self._default = default

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        out = np.full_like(labels, self._default)
        for src, dst in self._mapping.items():
            out[labels == src] = dst
        return out

    def __repr__(self) -> str:
        return f"RemapLabels(mapping={self._mapping}, default={self._default})"


class MaskLabels:
    """Set any label NOT in ``keep`` to ``ignore_value``.

    Useful for discarding rare or unknown classes before training.

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
        labels = np.asarray(labels).copy()
        mask = ~np.isin(labels, list(self._keep))
        labels[mask] = self._ignore
        return labels

    def __repr__(self) -> str:
        return f"MaskLabels(keep={sorted(self._keep)}, ignore_value={self._ignore})"
