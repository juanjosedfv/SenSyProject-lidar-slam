"""Backend trajectory adapters for the offline SLAM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slam_icp.io import read_trajectory_csv_compatible
from slam_icp.trajectory import TimedPose


@dataclass(frozen=True)
class BackendTrajectory:
    backend: str
    poses: list[TimedPose]
    trajectory_path: Path
    metadata: dict[str, Any]


def load_common_trajectory(
    trajectory_path: str | Path,
    expected_backend: str,
) -> BackendTrajectory:
    path = Path(trajectory_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Backend trajectory does not exist: {path}")
    poses = read_trajectory_csv_compatible(path)
    if not poses:
        raise ValueError(f"Backend trajectory contains no poses: {path}")
    for previous, current in zip(poses[:-1], poses[1:]):
        if current.timestamp <= previous.timestamp:
            raise ValueError("Backend trajectory timestamps must strictly increase")
    backends = {pose.backend for pose in poses}
    if backends != {expected_backend}:
        raise ValueError(
            f"Trajectory backend values {sorted(backends)} do not match {expected_backend}"
        )
    frames = {(pose.parent_frame, pose.child_frame) for pose in poses}
    if len(frames) != 1:
        raise ValueError("Backend trajectory contains inconsistent frame names")
    parent_frame, child_frame = next(iter(frames))
    return BackendTrajectory(
        backend=expected_backend,
        poses=poses,
        trajectory_path=path,
        metadata={
            "backend": expected_backend,
            "trajectory_path": str(path),
            "pose_count": len(poses),
            "first_timestamp": poses[0].timestamp,
            "last_timestamp": poses[-1].timestamp,
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "provenance": "existing common trajectory CSV; ICP was not rerun",
        },
    )


def load_baseline_icp_trajectory(path: str | Path) -> BackendTrajectory:
    return load_common_trajectory(path, "baseline_icp")


def load_kiss_icp_trajectory(path: str | Path) -> BackendTrajectory:
    return load_common_trajectory(path, "kiss_icp")


LOADERS = {
    "kiss_icp": load_kiss_icp_trajectory,
    "baseline_icp": load_baseline_icp_trajectory,
}

__all__ = ["LOADERS", "load_baseline_icp_trajectory", "load_kiss_icp_trajectory"]
