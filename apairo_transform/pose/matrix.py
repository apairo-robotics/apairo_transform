"""Pose / homogeneous-matrix transforms — applied at access time to pose arrays.

Two common conventions are supported:

* ``(N, 4, 4)`` or ``(4, 4)`` — already a homogeneous matrix.
* ``(N, 7)`` — ``[tx, ty, tz, qx, qy, qz, qw]`` (translation + quaternion).
* ``(N, 6)`` — ``[tx, ty, tz, rx, ry, rz]`` (translation + Euler ZYX, radians).

Typical usage::

    ds.transform("poses", PoseTo4x4())
    ds.transform("poses", InvertPose())
"""

from __future__ import annotations

import numpy as np


class PoseTo4x4:
    """Convert a pose array to a stack of 4x4 homogeneous matrices.

    Accepted input shapes:

    * ``(4, 4)`` or ``(N, 4, 4)`` — passed through unchanged.
    * ``(7,)`` or ``(N, 7)``     — ``[tx, ty, tz, qx, qy, qz, qw]``.
    * ``(6,)`` or ``(N, 6)``     — ``[tx, ty, tz, rx, ry, rz]`` (Euler ZYX, rad).
    """

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=np.float64)

        if poses.ndim == 2 and poses.shape == (4, 4):
            return poses
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            return poses

        single = poses.ndim == 1
        if single:
            poses = poses[None]  # (1, D)

        if poses.shape[1] == 7:
            result = _quat_to_4x4(poses)
        elif poses.shape[1] == 6:
            result = _euler_zyx_to_4x4(poses)
        else:
            raise ValueError(
                f"Cannot convert pose array with shape {poses.shape} to 4x4. "
                "Expected last dim 6 or 7."
            )

        return result[0] if single else result

    def __repr__(self) -> str:
        return "PoseTo4x4()"


class InvertPose:
    """Invert a 4x4 homogeneous transformation matrix (or a stack of them).

    Uses the closed-form inverse ``[R^T | -R^T t]`` which is exact for rigid
    transforms and avoids the numerical drift of ``np.linalg.inv``.

    Accepted input shapes: ``(4, 4)`` or ``(N, 4, 4)``.
    """

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=np.float64)

        if poses.shape == (4, 4):
            return _invert_single(poses)

        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            out = np.empty_like(poses)
            for i in range(len(poses)):
                out[i] = _invert_single(poses[i])
            return out

        raise ValueError(
            f"InvertPose expects shape (4, 4) or (N, 4, 4), got {poses.shape}"
        )

    def __repr__(self) -> str:
        return "InvertPose()"


# ── helpers ──────────────────────────────────────────────────────────────────

def _invert_single(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _quat_to_4x4(poses: np.ndarray) -> np.ndarray:
    """(N, 7) [tx ty tz qx qy qz qw] → (N, 4, 4)."""
    N = len(poses)
    tx, ty, tz = poses[:, 0], poses[:, 1], poses[:, 2]
    qx, qy, qz, qw = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]

    # normalise quaternions
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    out = np.zeros((N, 4, 4), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    out[:, 0, 1] = 2 * (qx * qy - qz * qw)
    out[:, 0, 2] = 2 * (qx * qz + qy * qw)
    out[:, 1, 0] = 2 * (qx * qy + qz * qw)
    out[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    out[:, 1, 2] = 2 * (qy * qz - qx * qw)
    out[:, 2, 0] = 2 * (qx * qz - qy * qw)
    out[:, 2, 1] = 2 * (qy * qz + qx * qw)
    out[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    out[:, 0, 3] = tx
    out[:, 1, 3] = ty
    out[:, 2, 3] = tz
    out[:, 3, 3] = 1.0
    return out


def _euler_zyx_to_4x4(poses: np.ndarray) -> np.ndarray:
    """(N, 6) [tx ty tz rx ry rz] ZYX Euler (rad) → (N, 4, 4)."""
    N = len(poses)
    tx, ty, tz = poses[:, 0], poses[:, 1], poses[:, 2]
    rx, ry, rz = poses[:, 3], poses[:, 4], poses[:, 5]

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    out = np.zeros((N, 4, 4), dtype=np.float64)
    out[:, 0, 0] = cy * cz
    out[:, 0, 1] = cz * sx * sy - cx * sz
    out[:, 0, 2] = cx * cz * sy + sx * sz
    out[:, 1, 0] = cy * sz
    out[:, 1, 1] = cx * cz + sx * sy * sz
    out[:, 1, 2] = cx * sy * sz - cz * sx
    out[:, 2, 0] = -sy
    out[:, 2, 1] = cy * sx
    out[:, 2, 2] = cx * cy
    out[:, 0, 3] = tx
    out[:, 1, 3] = ty
    out[:, 2, 3] = tz
    out[:, 3, 3] = 1.0
    return out
