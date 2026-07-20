"""Shared timestamped pose interchange format for all odometry backends."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CSV_FIELDS = (
    "timestamp",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "backend",
    "parent_frame",
    "child_frame",
)


@dataclass(frozen=True)
class TimedPose:
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    backend: str
    parent_frame: str
    child_frame: str


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return an ``(x, y, z, w)`` unit quaternion for a 3x3 rotation."""

    matrix = np.asarray(rotation, dtype="float64")
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype="float64")
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    """Return a 3x3 rotation for an ``(x, y, z, w)`` quaternion."""

    quaternion = np.array([qx, qy, qz, qw], dtype="float64")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype="float64")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype="float64",
    )


def timed_pose_to_transform(pose: TimedPose) -> np.ndarray:
    transform = np.eye(4, dtype="float64")
    transform[:3, :3] = quaternion_to_rotation_matrix(
        pose.qx, pose.qy, pose.qz, pose.qw
    )
    transform[:3, 3] = [pose.x, pose.y, pose.z]
    return transform


def from_odometry_steps(
    steps: Iterable,
    *,
    backend: str,
    parent_frame: str,
    child_frame: str,
) -> list[TimedPose]:
    """Convert native odometry steps at a backend boundary."""

    poses = []
    for step in steps:
        transform = np.asarray(step.transform, dtype="float64")
        if transform.shape != (4, 4):
            raise ValueError("odometry transforms must have shape (4, 4)")
        qx, qy, qz, qw = rotation_matrix_to_quaternion(transform[:3, :3])
        poses.append(
            TimedPose(
                timestamp=float(step.timestamp_s),
                x=float(transform[0, 3]),
                y=float(transform[1, 3]),
                z=float(transform[2, 3]),
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                backend=backend,
                parent_frame=parent_frame,
                child_frame=child_frame,
            )
        )
    return poses


def write_trajectory_csv(path: str | Path, poses: Iterable[TimedPose]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pose in poses:
            writer.writerow({field: getattr(pose, field) for field in CSV_FIELDS})


def read_trajectory_csv(path: str | Path) -> list[TimedPose]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                f"unexpected trajectory schema {reader.fieldnames}; expected {list(CSV_FIELDS)}"
            )
        return [
            TimedPose(
                timestamp=float(row["timestamp"]),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
                qw=float(row["qw"]),
                backend=row["backend"],
                parent_frame=row["parent_frame"],
                child_frame=row["child_frame"],
            )
            for row in reader
        ]


def trajectory_xyz(trajectory: Iterable) -> np.ndarray:
    return np.array([step.transform[:3, 3] for step in trajectory], dtype="float64")
