"""Tensorize apairo samples at the edge of the pipeline (PyTorch).

These are **terminal** helpers: they return ``torch`` tensors, so they break
apairo's numpy-in / numpy-out contract and must be the *last* hop -- a
``DataLoader``'s ``collate_fn`` -- never a mid-pipeline ``transform``. An apairo
dataset is already a map-style ``torch`` ``Dataset`` (it has ``__len__`` /
``__getitem__``), so there is nothing to wrap: pass it straight to a
``DataLoader`` and hand it one of these as the collate.

They cover the dense case. Sparse or padded batching of variable-length point
clouds (a batch-index vector, a ragged pad) is domain policy and stays your own
collate; :func:`to_torch_collate` falls back to a list of tensors there rather
than guess.

``torch`` is an optional extra: ``pip install apairo-transform[torch]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apairo.core.sample import Sample


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "to_torch_collate / to_tensors require PyTorch. Install it with "
            "`pip install apairo-transform[torch]` or `pip install torch`."
        ) from exc
    return torch


def to_tensors(sample: Sample) -> dict[str, Any]:
    """Every channel of one :class:`~apairo.core.sample.Sample` as a torch tensor.

    Terminal: use it inside a ``collate_fn`` (or after indexing), never as a
    chainable ``transform`` -- the result is tensors, not numpy."""
    torch = _torch()
    return {k: torch.as_tensor(np.asarray(v)) for k, v in sample.data.items()}


def to_torch_collate(batch: list[Sample]) -> dict[str, Any]:
    """Collate a list of Samples into ``{channel: tensor}`` for a ``DataLoader``.

    Per channel: stack into a ``(B, ...)`` tensor when every sample shares that
    channel's shape; otherwise return a ``list`` of tensors -- variable-length
    channels (raw point clouds) cannot dense-stack, and choosing a sparse/padded
    representation is the caller's policy. Terminal (returns tensors): the last
    hop into the ``DataLoader``, never a chainable transform.

        loader = DataLoader(ds.synchronize(...), batch_size=8,
                            collate_fn=to_torch_collate)
    """
    torch = _torch()
    if not batch:
        return {}
    out: dict[str, Any] = {}
    for key in batch[0].data:
        tensors = [torch.as_tensor(np.asarray(s.data[key])) for s in batch]
        uniform = len({tuple(t.shape) for t in tensors}) == 1
        out[key] = torch.stack(tensors) if uniform else tensors
    return out
