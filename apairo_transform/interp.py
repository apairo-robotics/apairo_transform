"""Value-level interpolators for ``ds.synchronize()``.

These implement the :class:`apairo.Interpolator` contract: synthesize a
channel value at the reference instant from its two bracketing events.
Use them for continuous signals only -- poses, IMU, commands -- never for
point clouds or images.

Typical usage::

    from apairo_transform.interp import LinearInterp, Se3Interp

    ds_sync = ds.synchronize(
        reference="velodyne_0",
        method={
            "gicp_poses": Se3Interp(),     # slerp rotation + lerp translation
            "cmd":        LinearInterp(),  # plain linear blend
        },                                  # unlisted channels -> "latest"
    )
"""

from __future__ import annotations

import numpy as np

from apairo import Interpolator


class LinearInterp(Interpolator):
    """Linear interpolation between the two bracketing event values.

    Works on any array shape that supports scalar blending (commands,
    velocities, IMU readings, scalar signals).  Do **not** use on rotations
    or quaternions -- use :class:`Se3Interp` instead.
    """

    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        return (1.0 - a) * np.asarray(v0) + a * np.asarray(v1)

    def __repr__(self) -> str:
        return "LinearInterp()"


class Se3Interp(Interpolator):
    """SE(3) pose interpolation: lerp on translation, slerp on rotation.

    Accepted formats (returned unchanged):

    * ``(7,)``   -- ``[tx, ty, tz, qx, qy, qz, qw]`` (translation + quaternion)
    * ``(4, 4)`` -- homogeneous transformation matrix

    Quaternions are interpolated along the shortest path (sign-corrected
    slerp), so ``q`` and ``-q`` inputs give identical results.
    """

    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        v0 = np.asarray(v0, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64)
        if v0.shape != v1.shape:
            raise ValueError(
                f"Bracketing poses have different shapes: {v0.shape} vs {v1.shape}"
            )

        if v0.shape == (7,):
            trans = (1.0 - a) * v0[:3] + a * v1[:3]
            quat = _slerp(v0[3:], v1[3:], a)
            return np.concatenate([trans, quat])

        if v0.shape == (4, 4):
            trans = (1.0 - a) * v0[:3, 3] + a * v1[:3, 3]
            quat = _slerp(_rot_to_quat(v0[:3, :3]), _rot_to_quat(v1[:3, :3]), a)
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = _quat_to_rot(quat)
            out[:3, 3] = trans
            return out

        raise ValueError(
            f"Se3Interp expects shape (7,) [tx ty tz qx qy qz qw] or (4, 4), "
            f"got {v0.shape}"
        )

    def __repr__(self) -> str:
        return "Se3Interp()"


# ── quaternion helpers ([qx, qy, qz, qw] convention, as in pose.matrix) ──────

def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-path spherical interpolation between two unit quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # q and -q encode the same rotation: take the short way
        q1 = -q1
        dot = -dot

    if dot > 0.9995:  # nearly parallel: nlerp is exact enough and stable
        out = (1.0 - a) * q0 + a * q1
        return out / np.linalg.norm(out)

    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)
    return (
        np.sin((1.0 - a) * omega) / sin_omega * q0
        + np.sin(a * omega) / sin_omega * q1
    )


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> [qx, qy, qz, qw] (Shepperd's method)."""
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] -> (3, 3) rotation matrix."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )
