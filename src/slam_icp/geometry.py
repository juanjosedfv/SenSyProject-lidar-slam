"""Backend-neutral geometry helpers using the project pose convention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .trajectory import TimedPose, quaternion_to_rotation_matrix


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ValueError("Quaternion has zero length")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _value(pose: TimedPose | Mapping[str, Any], name: str) -> Any:
    return pose[name] if isinstance(pose, Mapping) else getattr(pose, name)


def pose_to_transform(pose: TimedPose | Mapping[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(
        float(_value(pose, "qx")),
        float(_value(pose, "qy")),
        float(_value(pose, "qz")),
        float(_value(pose, "qw")),
    )
    transform[:3, 3] = [
        float(_value(pose, "x")),
        float(_value(pose, "y")),
        float(_value(pose, "z")),
    ]
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("Transform must have shape (4, 4)")
    u, _, vt = np.linalg.svd(value[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ value[:3, 3]
    return inverse


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    homogeneous = np.column_stack((values, np.ones(len(values))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def yaw_from_transform(transform: np.ndarray) -> float:
    value = np.asarray(transform, dtype=np.float64)
    return math.atan2(value[1, 0], value[0, 0])
