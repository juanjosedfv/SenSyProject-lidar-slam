"""Shared IO helpers for the offline SLAM pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from baseline_icp.odometry import write_ascii_pcd
from kiss_icp.adapter import find_bag_db as find_rosbag_database
from mapping import VoxelAccumulator, cloud_to_xyz, nearest_pose_index, read_cloud_index
from slam_icp.trajectory import CSV_FIELDS, TimedPose, read_trajectory_csv


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def read_trajectory_csv_compatible(path: str | Path) -> list[TimedPose]:
    """Read canonical fields while allowing documented result columns."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError(f"Trajectory CSV has no header: {input_path}")
        if fields == list(CSV_FIELDS):
            return read_trajectory_csv(input_path)
        missing = set(CSV_FIELDS) - set(fields)
        if missing:
            raise ValueError(
                "Trajectory CSV is missing canonical columns: "
                + ", ".join(sorted(missing))
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


def load_ascii_pcd(*args, **kwargs):
    from slam_icp.outputs import load_ascii_pcd as load

    return load(*args, **kwargs)


def load_nearest_pointcloud(*args, **kwargs):
    """Load a scan while keeping ROS imports out of non-ROS commands."""

    from slam_icp.loop_closure import load_nearest_pointcloud as load

    return load(*args, **kwargs)


def pointcloud2_to_xyz_for_registration(message):
    """Decode registration scans with the preserved field-order behavior."""

    from slam_icp.loop_closure import pointcloud2_to_xyz

    return pointcloud2_to_xyz(message)


def filter_points_by_range(*args, **kwargs):
    from slam_icp.loop_closure import filter_points_by_range as filter_points

    return filter_points(*args, **kwargs)


def load_keyframe_by_id(*args, **kwargs):
    from slam_icp.loop_closure import load_keyframe_by_id as load

    return load(*args, **kwargs)


__all__ = [
    "VoxelAccumulator",
    "cloud_to_xyz",
    "find_rosbag_database",
    "filter_points_by_range",
    "load_ascii_pcd",
    "load_nearest_pointcloud",
    "load_keyframe_by_id",
    "nearest_pose_index",
    "pointcloud2_to_xyz_for_registration",
    "read_cloud_index",
    "read_trajectory_csv_compatible",
    "write_ascii_pcd",
    "write_json",
]
