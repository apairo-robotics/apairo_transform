"""The random generator of a random transform, safe across worker processes.

A transform that builds ``np.random.default_rng(seed)`` in its constructor
hands every ``DataLoader`` worker a copy of the same generator state: all
workers draw the same "random" numbers, and since workers are recreated from
the unchanged parent, every epoch repeats them. :class:`Rng` builds the
generator where it is used instead, once per process and per worker epoch.
"""

from __future__ import annotations

import os
import sys

import numpy as np

__all__ = ["Rng"]


def _worker_seed() -> int | None:
    """The seed PyTorch gave this DataLoader worker, or ``None`` outside one.
    It differs between workers and between epochs, and follows
    ``torch.manual_seed``. Read only if torch is already imported."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        info = torch.utils.data.get_worker_info()
    except Exception:
        return None
    return None if info is None else int(info.seed)


class Rng:
    """A numpy ``Generator`` created on first use in each process.

    Args:
        seed: ``None`` for fresh entropy. With a seed, the stream is
            reproducible: the same seed in the main process gives the same
            draws, and inside a ``DataLoader`` worker the seed is combined
            with the worker's own, so workers and epochs differ while a run
            with ``torch.manual_seed`` set is reproducible end to end.

    It forwards the generator's methods (``uniform``, ``normal``, ``random``,
    ``choice``, ``permutation``...).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._generator: np.random.Generator | None = None
        self._context: tuple | None = None

    def get_config(self) -> dict:
        """What rebuilds it (the generator's state is not part of a recipe)."""
        return {"seed": self.seed}

    @property
    def generator(self) -> np.random.Generator:
        context = (os.getpid(), _worker_seed())
        if self._generator is None or context != self._context:
            worker = context[1]
            if self.seed is None:
                entropy = None  # fresh in every process
            elif worker is None:
                entropy = self.seed
            else:
                entropy = [int(self.seed), worker]
            self._generator = np.random.default_rng(entropy)
            self._context = context
        return self._generator

    def __getattr__(self, name: str):
        if name.startswith("_"):  # not a Generator method; also during unpickling
            raise AttributeError(name)
        return getattr(self.generator, name)

    def __getstate__(self) -> dict:
        # A copy sent to another process starts its own stream.
        return {"seed": self.seed, "_generator": None, "_context": None}

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed})"
