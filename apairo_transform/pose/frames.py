"""Resolve the static-transform tree that apairo exposes as ``ds.calibration``.

A robot's fixed sensor mounts form a tree of static edges, which apairo
*describes* (``calibration.yaml`` / ``ds.calibration``) without applying. The one
canonical operation on that tree -- resolving the transform between two frames --
lives on the calibration object itself; applying the result is a separate, data-
dependent op (points vs poses vs normals)::

    from apairo_transform import TransformPoints

    T = ds.calibration.get_tf("os_lidar", "base_link")  # T_base_link_from_os_lidar
    ds.transform("lidar", TransformPoints(T))           # apply it to the "lidar" channel

:func:`lookup_transform` is the standalone form for a raw
``{"<parent>_to_<child>": 4x4}`` dict.
"""
from __future__ import annotations

import warnings

import numpy as np

from apairo.core.config import Calibration


def lookup_transform(
    calibration: dict[str, np.ndarray], target: str, source: str
) -> np.ndarray:
    """Deprecated -- use ``ds.calibration.get_tf(source, target)``.

    Resolution moved onto the calibration object itself
    (:meth:`apairo.core.config.Calibration.get_tf`). This wrapper stays for
    back-compat; mind that ``get_tf`` takes ``(source, target)`` -- the opposite
    of this function's ROS ``lookupTransform(target, source)`` order.
    """
    warnings.warn(
        "lookup_transform(cal, target, source) is deprecated; use "
        "ds.calibration.get_tf(source, target) -- or Calibration(cal).get_tf(...) "
        "for a raw dict. Note the argument order flips to (source, target).",
        DeprecationWarning,
        stacklevel=2,
    )
    return Calibration(calibration).get_tf(source, target)
