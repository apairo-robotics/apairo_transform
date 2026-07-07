"""Image transforms — applied at access time to ``(H, W, C)`` arrays.

Operate on the ``(H, W, 3)`` uint8 images an apairo dataset yields (e.g. from a
``TarImageLoader``) and return numpy arrays, composable with
``apairo.core.transform.Compose``.

Typical usage::

    ds.transform("image", Compose([Resize(224, 224), ToFloat(), Normalize(), ToCHW()]))

``Resize`` needs OpenCV (install the ``image`` extra:
``pip install apairo-transform[image]``); the other transforms are pure numpy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# OpenCV interpolation flags, resolved lazily so importing this module (and the
# pure-numpy transforms below) never requires OpenCV.
_INTERP_NAMES = ("area", "linear", "nearest", "cubic")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Resize:
    """Resize an image to ``(height, width)``.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
        method: Interpolation — one of ``"area"`` (default, best for
            downsampling), ``"linear"``, ``"nearest"``, ``"cubic"``.

    Requires OpenCV (the ``image`` extra).
    """

    def __init__(self, width: int, height: int, method: str = "area") -> None:
        if method not in _INTERP_NAMES:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(_INTERP_NAMES)}"
            )
        self.width = width
        self.height = height
        self.method = method

    def __call__(self, img: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Resize requires OpenCV. Install it with "
                "`pip install apairo-transform[image]` or `pip install opencv-python`."
            ) from exc
        interp = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
        }[self.method]
        return cv2.resize(
            img, (self.width, self.height), interpolation=interp
        ).astype(np.uint8)

    def __repr__(self) -> str:
        return f"Resize({self.width}x{self.height}, method={self.method!r})"


class CenterCrop:
    """Crop ``(crop_h, crop_w)`` from the centre of the image."""

    def __init__(self, crop_h: int, crop_w: int) -> None:
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        y = max(0, (h - self.crop_h) // 2)
        x = max(0, (w - self.crop_w) // 2)
        return img[y : y + self.crop_h, x : x + self.crop_w]

    def __repr__(self) -> str:
        return f"CenterCrop({self.crop_h}x{self.crop_w})"


class ToFloat:
    """Convert uint8 ``[0, 255]`` → float32 ``[0.0, 1.0]``."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0

    def __repr__(self) -> str:
        return "ToFloat()"


class Normalize:
    """Per-channel normalisation ``(x - mean) / std``.

    Expects float32 input (apply :class:`ToFloat` first). Defaults to ImageNet
    statistics.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = _IMAGENET_MEAN,
        std: Tuple[float, float, float] = _IMAGENET_STD,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std

    def __repr__(self) -> str:
        return (
            f"Normalize(mean={tuple(self.mean.flatten().tolist())}, "
            f"std={tuple(self.std.flatten().tolist())})"
        )


class ToCHW:
    """``(H, W, C)`` → ``(C, H, W)`` — channel-first layout for PyTorch."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(img.transpose(2, 0, 1))

    def __repr__(self) -> str:
        return "ToCHW()"
