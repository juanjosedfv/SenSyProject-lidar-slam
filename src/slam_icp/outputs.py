from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation import (
    align_2d_rigid,
    compute_rmse,
    gnss_to_enu,
    interpolate_ground_truth,
    load_gnss_ground_truth,
    origin_from_gnss,
)
from slam_icp.geometry import pose_to_transform
from slam_icp.io import (
    VoxelAccumulator,
    cloud_to_xyz,
    find_rosbag_database,
    nearest_pose_index,
    read_cloud_index,
    write_ascii_pcd,
)
from level import (
    apply_rotation,
    describe,
    level_rotation,
    level_transform,
)
from preprocess import filter_points
from slam_icp.pose_graph import load_full_trajectory

def load_trajectory_transforms(
    trajectory_path: str | Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    poses = load_full_trajectory(
        trajectory_path
    )

    return trajectory_transforms(poses)


def trajectory_transforms(
    poses: list[dict],
) -> tuple[np.ndarray, list[np.ndarray]]:

    pose_times = np.asarray(
        [
            pose["timestamp"]
            for pose in poses
        ],
        dtype=np.float64,
    )

    transforms = [
        pose_to_transform(pose)
        for pose in poses
    ]

    order = np.argsort(
        pose_times,
        kind="stable",
    )

    sorted_times = pose_times[order]

    sorted_transforms = [
        transforms[index]
        for index in order
    ]

    return sorted_times, sorted_transforms


def calculate_common_level_transform(
    corrected_pose_times: np.ndarray,
    reference_trajectory_path: str | Path,
) -> tuple[np.ndarray, dict]:
    reference_poses = load_full_trajectory(
        reference_trajectory_path
    )

    return calculate_common_level_transform_from_poses(
        corrected_pose_times,
        reference_poses,
    )


def calculate_common_level_transform_from_poses(
    corrected_pose_times: np.ndarray,
    reference_poses: list[dict],
) -> tuple[np.ndarray, dict]:

    reference_times = np.asarray(
        [
            pose["timestamp"]
            for pose in reference_poses
        ],
        dtype=np.float64,
    )

    if len(reference_times) != len(
        corrected_pose_times
    ):
        raise ValueError(
            "The leveling reference and corrected "
            "trajectory have different pose counts"
        )

    if not np.allclose(
        reference_times,
        corrected_pose_times,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The leveling reference and corrected "
            "trajectory timestamps do not match"
        )

    reference_xyz = np.asarray(
        [
            [
                pose["x"],
                pose["y"],
                pose["z"],
            ]
            for pose in reference_poses
        ],
        dtype=np.float64,
    )

    return level_transform(
        reference_xyz
    )


def write_map_summary(
    path: str | Path,
    summary: dict,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_slam_map(
    trajectory_path: str | Path,
    lidar_bag_path: str | Path,
    lidar_topic: str,
    output_pcd_path: str | Path,
    scan_stride: int,
    voxel_size_m: float,
    minimum_range_m: float,
    maximum_range_m: float,
    pose_tolerance_s: float,
    level_enabled: bool,
    level_reference_trajectory_path: (
        str | Path | None
    ),
    trajectory_poses: list[dict] | None = None,
    level_reference_poses: list[dict] | None = None,
    summary_path: str | Path | None = None,
) -> dict:
    if scan_stride < 1:
        raise ValueError(
            "Scan stride must be at least one"
        )

    if voxel_size_m <= 0.0:
        raise ValueError(
            "Voxel size must be greater than zero"
        )

    if minimum_range_m < 0.0:
        raise ValueError(
            "Minimum range cannot be negative"
        )

    if maximum_range_m <= minimum_range_m:
        raise ValueError(
            "Maximum range must be greater "
            "than minimum range"
        )

    if pose_tolerance_s < 0.0:
        raise ValueError(
            "Pose tolerance cannot be negative"
        )

    trajectory_path = Path(
        trajectory_path
    ).expanduser()

    lidar_bag_path = Path(
        lidar_bag_path
    ).expanduser()

    output_pcd_path = Path(
        output_pcd_path
    ).expanduser()

    if trajectory_poses is None and not trajectory_path.exists():
        raise FileNotFoundError(
            f"SLAM trajectory not found: "
            f"{trajectory_path}"
        )

    if not lidar_bag_path.exists():
        raise FileNotFoundError(
            f"LiDAR rosbag not found: "
            f"{lidar_bag_path}"
        )

    try:
        from rclpy.serialization import (
            deserialize_message,
        )
        from sensor_msgs.msg import PointCloud2
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 must be sourced before "
            "reading PointCloud2 messages"
        ) from error

    pose_times, transforms = (
        trajectory_transforms(trajectory_poses)
        if trajectory_poses is not None
        else load_trajectory_transforms(trajectory_path)
    )

    level_information = None

    if level_enabled:
        if (
            level_reference_trajectory_path
            is None
        ):
            raise ValueError(
                "A leveling reference trajectory "
                "is required when --level is used"
            )

        if level_reference_poses is not None:
            levelling_transform, level_information = (
                calculate_common_level_transform_from_poses(
                    corrected_pose_times=pose_times,
                    reference_poses=level_reference_poses,
                )
            )
        else:
            levelling_transform, level_information = (
                calculate_common_level_transform(
                    corrected_pose_times=pose_times,
                    reference_trajectory_path=level_reference_trajectory_path,
                )
            )

        transforms = [
            levelling_transform @ transform
            for transform in transforms
        ]

    database_path = find_rosbag_database(
        lidar_bag_path
    )

    message_ids = read_cloud_index(
        database_path,
        lidar_topic,
    )

    selected_message_ids = message_ids[
        ::scan_stride
    ]

    print("Building SLAM point-cloud map")
    print(
        f"  trajectory : {trajectory_path}"
    )
    print(
        f"  LiDAR bag  : {lidar_bag_path}"
    )
    print(
        f"  topic      : {lidar_topic}"
    )
    print(
        f"  poses      : {len(pose_times)}"
    )
    print(
        f"  pose range : "
        f"{pose_times[0]:.6f} .. "
        f"{pose_times[-1]:.6f}"
    )
    print(
        f"  scans      : {len(message_ids)} total"
    )
    print(
        f"  selected   : "
        f"{len(selected_message_ids)} "
        f"(every {scan_stride}th scan)"
    )
    print(
        f"  range gate : "
        f"{minimum_range_m} .. "
        f"{maximum_range_m} m"
    )
    print(
        f"  voxel      : {voxel_size_m} m"
    )
    print(
        f"  tolerance  : "
        f"{pose_tolerance_s} s"
    )

    if level_information is not None:
        print(
            "  gravity alignment: "
            f"{describe(level_information)}"
        )

    voxel_grid = VoxelAccumulator(
        voxel_size_m
    )

    used_scans = 0
    skipped_scans = 0
    empty_scans = 0
    total_filtered_points = 0

    scan_pose_differences: list[float] = []

    start_time = time.time()

    with sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    ) as connection:
        for selected_index, message_id in enumerate(
            selected_message_ids,
            start=1,
        ):
            row = connection.execute(
                """
                SELECT data
                FROM messages
                WHERE id = ?
                """,
                (int(message_id),),
            ).fetchone()

            if row is None:
                skipped_scans += 1
                continue

            message = deserialize_message(
                bytes(row[0]),
                PointCloud2,
            )

            scan_timestamp = (
                float(message.header.stamp.sec)
                + float(
                    message.header.stamp.nanosec
                )
                * 1e-9
            )

            pose_index = nearest_pose_index(
                pose_times,
                scan_timestamp,
                pose_tolerance_s,
            )

            if pose_index is None:
                skipped_scans += 1
                continue

            scan_pose_dt = abs(
                pose_times[pose_index]
                - scan_timestamp
            )

            scan_pose_differences.append(
                scan_pose_dt
            )

            points = cloud_to_xyz(
                message
            )

            points = filter_points(
                points,
                min_range_m=minimum_range_m,
                max_range_m=maximum_range_m,
                remove_ground=False,
                ground_z_threshold_m=-1.5,
            )

            if len(points) == 0:
                empty_scans += 1
                continue

            transform = transforms[
                pose_index
            ]

            world_points = (
                points @ transform[:3, :3].T
                + transform[:3, 3]
            )

            voxel_grid.add(
                world_points
            )

            used_scans += 1

            total_filtered_points += len(
                points
            )

            if (
                selected_index == 1
                or selected_index % 25 == 0
                or selected_index
                == len(selected_message_ids)
            ):
                elapsed_seconds = (
                    time.time() - start_time
                )

                print(
                    f"  {selected_index}/"
                    f"{len(selected_message_ids)} "
                    f"scans | "
                    f"{len(voxel_grid):,} voxels | "
                    f"{elapsed_seconds:.0f}s"
                )

    if used_scans == 0:
        raise RuntimeError(
            "No LiDAR scan was successfully "
            "matched to a SLAM pose"
        )

    map_points = voxel_grid.points()

    map_minimum = np.min(
        map_points,
        axis=0,
    )

    map_maximum = np.max(
        map_points,
        axis=0,
    )

    map_extent = (
        map_maximum - map_minimum
    )

    output_pcd_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_ascii_pcd(
        output_pcd_path,
        map_points,
    )

    summary_path = (
        Path(summary_path).expanduser()
        if summary_path is not None
        else output_pcd_path.parent
        / f"{output_pcd_path.stem}_summary.json"
    )

    summary = {
        "trajectory": str(trajectory_path),
        "lidar_bag": str(lidar_bag_path),
        "lidar_topic": lidar_topic,
        "output_pcd": str(output_pcd_path),
        "pose_count": len(pose_times),
        "total_lidar_scans": len(message_ids),
        "selected_lidar_scans": len(
            selected_message_ids
        ),
        "used_lidar_scans": used_scans,
        "skipped_lidar_scans": skipped_scans,
        "empty_lidar_scans": empty_scans,
        "total_filtered_points": (
            total_filtered_points
        ),
        "final_voxel_count": len(
            map_points
        ),
        "scan_stride": scan_stride,
        "voxel_size_m": voxel_size_m,
        "minimum_range_m": minimum_range_m,
        "maximum_range_m": maximum_range_m,
        "pose_tolerance_s": pose_tolerance_s,
        "leveling_enabled": level_enabled,
        "level_reference_trajectory": (
            str(level_reference_trajectory_path)
            if level_reference_trajectory_path
            is not None
            else None
        ),
        "mean_scan_pose_dt_s": float(
            np.mean(scan_pose_differences)
        ),
        "maximum_scan_pose_dt_s": float(
            np.max(scan_pose_differences)
        ),
        "map_minimum_xyz_m": (
            map_minimum.tolist()
        ),
        "map_maximum_xyz_m": (
            map_maximum.tolist()
        ),
        "map_extent_xyz_m": (
            map_extent.tolist()
        ),
        "pcd_size_mb": (
            output_pcd_path.stat().st_size
            / 1e6
        ),
    }

    write_map_summary(
        summary_path,
        summary,
    )

    summary["summary_path"] = str(
        summary_path
    )

    return summary


# SLAM evaluation helpers consolidated from the development module.

def trajectory_to_arrays(
    poses: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(
        [
            pose["timestamp"]
            for pose in poses
        ],
        dtype=np.float64,
    )

    xy = np.asarray(
        [
            [
                pose["x"],
                pose["y"],
            ]
            for pose in poses
        ],
        dtype=np.float64,
    )

    return timestamps, xy

def trajectory_to_xyz(
    poses: list[dict],
) -> np.ndarray:
    return np.asarray(
        [
            [
                pose["x"],
                pose["y"],
                pose["z"],
            ]
            for pose in poses
        ],
        dtype=np.float64,
    )


def validate_trajectory_pair(
    original_poses: list[dict],
    slam_poses: list[dict],
) -> None:
    if len(original_poses) != len(slam_poses):
        raise ValueError(
            "Original and SLAM trajectories have "
            "different pose counts: "
            f"{len(original_poses)} and "
            f"{len(slam_poses)}"
        )

    original_timestamps, _ = trajectory_to_arrays(
        original_poses
    )

    slam_timestamps, _ = trajectory_to_arrays(
        slam_poses
    )

    timestamps_match = np.allclose(
        original_timestamps,
        slam_timestamps,
        rtol=0.0,
        atol=1e-6,
    )

    if not timestamps_match:
        raise ValueError(
            "Original and SLAM trajectory "
            "timestamps do not match"
        )


def path_length_xy(
    xy: np.ndarray,
) -> float:
    if len(xy) < 2:
        return 0.0

    position_changes = np.diff(
        xy,
        axis=0,
    )

    segment_lengths = np.linalg.norm(
        position_changes,
        axis=1,
    )

    return float(
        np.sum(segment_lengths)
    )


def endpoint_displacement_xy(
    xy: np.ndarray,
) -> float:
    if len(xy) < 2:
        return 0.0

    displacement = (
        xy[-1]
        - xy[0]
    )

    return float(
        np.linalg.norm(displacement)
    )


def evaluate_one_trajectory(
    trajectory_xy: np.ndarray,
    reference_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    estimated_valid = trajectory_xy[
        valid_mask
    ]

    reference_valid = reference_xy[
        valid_mask
    ]

    aligned_valid, rotation, translation = (
        align_2d_rigid(
            estimated_valid,
            reference_valid,
        )
    )

    aligned_all = (
        trajectory_xy
        @ rotation.T
        + translation
    )

    error_xy = (
        aligned_valid
        - reference_valid
    )

    horizontal_error_m = np.linalg.norm(
        error_xy,
        axis=1,
    )

    return {
        "aligned_all": aligned_all,
        "aligned_valid": aligned_valid,
        "horizontal_error_m": (
            horizontal_error_m
        ),
        "rmse_m": compute_rmse(
            error_xy
        ),
        "mean_error_m": float(
            np.mean(horizontal_error_m)
        ),
        "median_error_m": float(
            np.median(horizontal_error_m)
        ),
        "maximum_error_m": float(
            np.max(horizontal_error_m)
        ),
        "rotation": rotation,
        "translation": translation,
    }


def write_evaluation_csv(
    path: str | Path,
    timestamps: np.ndarray,
    valid_mask: np.ndarray,
    reference_xy: np.ndarray,
    original_result: dict,
    slam_result: dict,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_timestamps = timestamps[
        valid_mask
    ]

    valid_reference = reference_xy[
        valid_mask
    ]

    original_aligned = original_result[
        "aligned_valid"
    ]

    slam_aligned = slam_result[
        "aligned_valid"
    ]

    original_errors = original_result[
        "horizontal_error_m"
    ]

    slam_errors = slam_result[
        "horizontal_error_m"
    ]

    fieldnames = [
        "timestamp",
        "gnss_east_m",
        "gnss_north_m",
        "kiss_east_m",
        "kiss_north_m",
        "slam_east_m",
        "slam_north_m",
        "kiss_error_m",
        "slam_error_m",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            len(valid_timestamps)
        ):
            writer.writerow(
                {
                    "timestamp": (
                        valid_timestamps[index]
                    ),
                    "gnss_east_m": (
                        valid_reference[index, 0]
                    ),
                    "gnss_north_m": (
                        valid_reference[index, 1]
                    ),
                    "kiss_east_m": (
                        original_aligned[index, 0]
                    ),
                    "kiss_north_m": (
                        original_aligned[index, 1]
                    ),
                    "slam_east_m": (
                        slam_aligned[index, 0]
                    ),
                    "slam_north_m": (
                        slam_aligned[index, 1]
                    ),
                    "kiss_error_m": (
                        original_errors[index]
                    ),
                    "slam_error_m": (
                        slam_errors[index]
                    ),
                }
            )


def plot_trajectory_comparison(
    path: str | Path,
    reference_xy: np.ndarray,
    valid_mask: np.ndarray,
    original_result: dict,
    slam_result: dict,
    original_label: str = "KISS",
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_valid = reference_xy[
        valid_mask
    ]

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.plot(
        reference_valid[:, 0],
        reference_valid[:, 1],
        linewidth=2.0,
        label="Corrected GNSS",
    )

    axis.plot(
        original_result[
            "aligned_valid"
        ][:, 0],
        original_result[
            "aligned_valid"
        ][:, 1],
        linewidth=1.2,
        label=f"Original {original_label}",
    )

    axis.plot(
        slam_result[
            "aligned_valid"
        ][:, 0],
        slam_result[
            "aligned_valid"
        ][:, 1],
        linewidth=1.5,
        label="SLAM optimized",
    )

    axis.scatter(
        [reference_valid[0, 0]],
        [reference_valid[0, 1]],
        s=70,
        marker="o",
        label="Start",
    )

    axis.scatter(
        [reference_valid[-1, 0]],
        [reference_valid[-1, 1]],
        s=70,
        marker="x",
        label="GNSS end",
    )

    axis.set_title(
        f"{original_label} and SLAM trajectories "
        "compared with corrected GNSS"
    )

    axis.set_xlabel(
        "East [m]"
    )

    axis.set_ylabel(
        "North [m]"
    )

    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def plot_error_comparison(
    path: str | Path,
    timestamps: np.ndarray,
    valid_mask: np.ndarray,
    original_result: dict,
    slam_result: dict,
    original_label: str = "KISS",
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_timestamps = timestamps[
        valid_mask
    ]

    relative_time_s = (
        valid_timestamps
        - valid_timestamps[0]
    )

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        relative_time_s,
        original_result[
            "horizontal_error_m"
        ],
        linewidth=1.0,
        label=(
            f"Original {original_label} error "
            f"(RMSE {original_result['rmse_m']:.3f} m)"
        ),
    )

    axis.plot(
        relative_time_s,
        slam_result[
            "horizontal_error_m"
        ],
        linewidth=1.0,
        label=f"SLAM error (RMSE {slam_result['rmse_m']:.3f} m)",
    )

    axis.set_title(
        "Horizontal position error "
        "before and after SLAM"
    )

    axis.set_xlabel(
        "Time since first GNSS match [s]"
    )

    axis.set_ylabel(
        "Horizontal error [m]"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def write_summary_json(
    path: str | Path,
    summary: dict,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_slam(
    original_trajectory_path: str | Path,
    slam_trajectory_path: str | Path,
    gnss_csv_path: str | Path,
    gnss_time_offset_s: float,
    level_enabled: bool,
    output_directory: str | Path,
    original_poses: list[dict] | None = None,
    slam_poses: list[dict] | None = None,
    write_detailed_csv: bool = True,
    generate_plots: bool = True,
    original_label: str = "KISS",
    plot_directory: str | Path | None = None,
    trajectory_plot_name: str = "trajectory_kiss_slam_gnss.png",
    error_plot_name: str = "error_kiss_vs_slam.png",
) -> dict:
    if original_poses is None:
        original_poses = load_full_trajectory(
            original_trajectory_path
        )

    if slam_poses is None:
        slam_poses = load_full_trajectory(
            slam_trajectory_path
        )

    validate_trajectory_pair(
        original_poses,
        slam_poses,
    )

    timestamps, _ = trajectory_to_arrays(
        original_poses
    )

    original_xyz = trajectory_to_xyz(
        original_poses
    )

    slam_xyz = trajectory_to_xyz(
        slam_poses
    )

    level_information = None

    if level_enabled:
        level_rotation_matrix, level_information = (
            level_rotation(
                original_xyz
            )
        )

        original_xyz = apply_rotation(
            original_xyz,
            level_rotation_matrix,
        )

        slam_xyz = apply_rotation(
            slam_xyz,
            level_rotation_matrix,
        )

    original_xy = original_xyz[:, :2]
    slam_xy = slam_xyz[:, :2]

    gnss_raw = load_gnss_ground_truth(
        gnss_csv_path
    )

    gnss_origin = origin_from_gnss(
        gnss_raw
    )

    gnss_enu = gnss_to_enu(
        gnss_raw,
        gnss_origin,
    )

    reference = interpolate_ground_truth(
        gnss_df=gnss_enu,
        timestamps_s=timestamps,
        time_offset_s=gnss_time_offset_s,
    )

    valid_mask = np.asarray(
        reference["valid"],
        dtype=bool,
    )

    valid_count = int(
        np.sum(valid_mask)
    )

    if valid_count < 2:
        raise RuntimeError(
            "Fewer than two poses overlap "
            "with corrected GNSS"
        )

    reference_xy = np.column_stack(
        (
            reference["east_m"],
            reference["north_m"],
        )
    )

    original_result = (
        evaluate_one_trajectory(
            trajectory_xy=original_xy,
            reference_xy=reference_xy,
            valid_mask=valid_mask,
        )
    )

    slam_result = evaluate_one_trajectory(
        trajectory_xy=slam_xy,
        reference_xy=reference_xy,
        valid_mask=valid_mask,
    )

    gnss_valid_xy = reference_xy[
        valid_mask
    ]

    original_valid_xy = original_xy[
        valid_mask
    ]

    slam_valid_xy = slam_xy[
        valid_mask
    ]

    gnss_path_m = path_length_xy(
        gnss_valid_xy
    )

    original_full_path_m = path_length_xy(
        original_xy
    )

    slam_full_path_m = path_length_xy(
        slam_xy
    )

    original_overlap_path_m = path_length_xy(
        original_valid_xy
    )

    slam_overlap_path_m = path_length_xy(
        slam_valid_xy
    )

    original_endpoint_m = (
        endpoint_displacement_xy(
            original_valid_xy
        )
    )

    slam_endpoint_m = (
        endpoint_displacement_xy(
            slam_valid_xy
        )
    )

    gnss_endpoint_m = (
        endpoint_displacement_xy(
            gnss_valid_xy
        )
    )

    rmse_improvement_m = (
        original_result["rmse_m"]
        - slam_result["rmse_m"]
    )

    if original_result["rmse_m"] > 0.0:
        rmse_improvement_percent = (
            100.0
            * rmse_improvement_m
            / original_result["rmse_m"]
        )
    else:
        rmse_improvement_percent = 0.0

    summary = {
        "alignment_method": (
            "common 3D trajectory leveling followed by "
            "independent 2D rigid rotation and translation; "
            "no scale"
            if level_enabled
            else
            "independent 2D rigid rotation and translation; "
            "no scale"
        ),
        "leveling_enabled": level_enabled,
        "pose_count": len(original_poses),
        "gnss_match_count": valid_count,
        "gnss_time_offset_s": (
            gnss_time_offset_s
        ),
        "first_timestamp": float(
            timestamps[0]
        ),
        "last_timestamp": float(
            timestamps[-1]
        ),
        "original_kiss": {
            "horizontal_rmse_m": (
                original_result["rmse_m"]
            ),
            "mean_error_m": (
                original_result["mean_error_m"]
            ),
            "median_error_m": (
                original_result["median_error_m"]
            ),
            "maximum_error_m": (
                original_result["maximum_error_m"]
            ),
            "full_path_length_m": (
                original_full_path_m
            ),
            "gnss_overlap_path_length_m": (
                original_overlap_path_m
            ),
            "path_ratio_to_gnss": (
                original_full_path_m
                / gnss_path_m
                if gnss_path_m > 0.0
                else None
            ),
            "endpoint_displacement_m": (
                original_endpoint_m
            ),
        },
        "slam": {
            "horizontal_rmse_m": (
                slam_result["rmse_m"]
            ),
            "mean_error_m": (
                slam_result["mean_error_m"]
            ),
            "median_error_m": (
                slam_result["median_error_m"]
            ),
            "maximum_error_m": (
                slam_result["maximum_error_m"]
            ),
            "full_path_length_m": (
                slam_full_path_m
            ),
            "gnss_overlap_path_length_m": (
                slam_overlap_path_m
            ),
            "path_ratio_to_gnss": (
                slam_full_path_m
                / gnss_path_m
                if gnss_path_m > 0.0
                else None
            ),
            "endpoint_displacement_m": (
                slam_endpoint_m
            ),
        },
        "gnss": {
            "path_length_m": gnss_path_m,
            "endpoint_displacement_m": (
                gnss_endpoint_m
            ),
        },
        "rmse_improvement_m": (
            rmse_improvement_m
        ),
        "rmse_improvement_percent": (
            rmse_improvement_percent
        ),
    }

    summary["original_backend_name"] = original_label
    summary["original_backend"] = dict(summary["original_kiss"])

    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_path
        / "slam_evaluation.csv"
    )

    summary_path = (
        output_path
        / "slam_evaluation_summary.json"
    )

    plot_path = (
        Path(plot_directory).expanduser()
        if plot_directory is not None
        else output_path / "plots"
    )

    trajectory_plot_path = plot_path / trajectory_plot_name

    error_plot_path = plot_path / error_plot_name

    if write_detailed_csv:
        write_evaluation_csv(
            path=csv_path,
            timestamps=timestamps,
            valid_mask=valid_mask,
            reference_xy=reference_xy,
            original_result=original_result,
            slam_result=slam_result,
        )

    write_summary_json(
        summary_path,
        summary,
    )

    if generate_plots:
        plot_trajectory_comparison(
            path=trajectory_plot_path,
            reference_xy=reference_xy,
            valid_mask=valid_mask,
            original_result=original_result,
            slam_result=slam_result,
            original_label=original_label,
        )

        plot_error_comparison(
            path=error_plot_path,
            timestamps=timestamps,
            valid_mask=valid_mask,
            original_result=original_result,
            slam_result=slam_result,
            original_label=original_label,
        )

    summary["evaluation_csv"] = str(csv_path) if write_detailed_csv else None

    summary["summary_json"] = str(
        summary_path
    )

    summary["trajectory_plot"] = str(trajectory_plot_path) if generate_plots else None

    summary["error_plot"] = str(error_plot_path) if generate_plots else None

    return summary


# Map plotting helpers consolidated from the development module.

def load_ascii_pcd(path: str | Path) -> np.ndarray:
    """Load XYZ coordinates from an ASCII PCD file."""

    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"PCD file does not exist: {input_path}")

    fields = None
    with input_path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header is incomplete: {input_path}")
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0].upper() == "FIELDS":
                fields = tokens[1:]
            if tokens[0].upper() == "DATA":
                if len(tokens) != 2 or tokens[1].lower() != "ascii":
                    raise ValueError("Only ASCII PCD files are supported")
                break

        if fields is None:
            raise ValueError(f"PCD file has no FIELDS entry: {input_path}")
        missing = [name for name in ("x", "y", "z") if name not in fields]
        if missing:
            raise ValueError(f"PCD file is missing fields: {', '.join(missing)}")
        data = np.loadtxt(stream, dtype="float64", ndmin=2)

    columns = [fields.index(name) for name in ("x", "y", "z")]
    points = data[:, columns]
    return points[np.isfinite(points).all(axis=1)]


def filter_height(
    points: np.ndarray,
    minimum_z_m: float | None,
    maximum_z_m: float | None,
) -> np.ndarray:
    valid = np.ones(len(points), dtype=bool)
    if minimum_z_m is not None:
        valid &= points[:, 2] >= minimum_z_m
    if maximum_z_m is not None:
        valid &= points[:, 2] <= maximum_z_m
    return points[valid]


def trajectory_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    poses = load_full_trajectory(path)
    return trajectory_arrays_from_poses(poses)


def trajectory_arrays_from_poses(poses: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray([pose["timestamp"] for pose in poses], dtype="float64")
    xyz = np.asarray(
        [[pose["x"], pose["y"], pose["z"]] for pose in poses],
        dtype="float64",
    )
    return timestamps, xyz


def load_keyframe_position(path: str | Path, keyframe_id: int) -> np.ndarray:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["keyframe_id"]) == keyframe_id:
                return np.asarray(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype="float64",
                )
    raise ValueError(f"Keyframe {keyframe_id} was not found in {input_path}")


def keyframe_position(keyframes: list[dict], keyframe_id: int) -> np.ndarray:
    for keyframe in keyframes:
        if int(keyframe["keyframe_id"]) == keyframe_id:
            return np.asarray(
                [keyframe["x"], keyframe["y"], keyframe["z"]],
                dtype="float64",
            )
    raise ValueError(f"Keyframe {keyframe_id} was not found")


def crop_xy(points: np.ndarray, centre_xy: np.ndarray, radius_m: float) -> np.ndarray:
    distance = np.linalg.norm(points[:, :2] - centre_xy[:2], axis=1)
    return points[distance <= radius_m]


def shared_xy_limits(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    combined = np.vstack((first[:, :2], second[:, :2]))
    low = combined.min(axis=0)
    high = combined.max(axis=0)
    span = high - low
    margin = np.maximum(0.02 * span, 1.0)
    return (
        (float(low[0] - margin[0]), float(high[0] + margin[0])),
        (float(low[1] - margin[1]), float(high[1] + margin[1])),
    )


def shared_height_range(
    first: np.ndarray,
    second: np.ndarray,
    requested_minimum: float | None = None,
    requested_maximum: float | None = None,
) -> tuple[float, float]:
    minimum = (
        float(requested_minimum)
        if requested_minimum is not None
        else float(min(first[:, 2].min(), second[:, 2].min()))
    )
    maximum = (
        float(requested_maximum)
        if requested_maximum is not None
        else float(max(first[:, 2].max(), second[:, 2].max()))
    )
    if maximum <= minimum:
        maximum = minimum + 1e-6
    return minimum, maximum


def calculate_leveled_loop_centre(
    keyframes_path: str | Path,
    source_id: int,
    target_id: int,
    level_rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = load_keyframe_position(keyframes_path, source_id)
    target = load_keyframe_position(keyframes_path, target_id)
    source_leveled = apply_rotation(source[None, :], level_rotation_matrix)[0]
    target_leveled = apply_rotation(target[None, :], level_rotation_matrix)[0]
    centre = 0.5 * (source_leveled + target_leveled)
    return centre, source_leveled, target_leveled


def calculate_leveled_loop_centre_from_keyframes(
    keyframes: list[dict],
    source_id: int,
    target_id: int,
    level_rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = keyframe_position(keyframes, source_id)
    target = keyframe_position(keyframes, target_id)
    source_leveled = apply_rotation(source[None, :], level_rotation_matrix)[0]
    target_leveled = apply_rotation(target[None, :], level_rotation_matrix)[0]
    return 0.5 * (source_leveled + target_leveled), source_leveled, target_leveled


def align_route_to_gnss(
    route_xy: np.ndarray,
    reference_xy: np.ndarray,
    valid: np.ndarray,
) -> dict:
    aligned_valid, rotation, translation = align_2d_rigid(
        route_xy[valid], reference_xy[valid]
    )
    return {
        "aligned_valid": aligned_valid,
        "aligned_all": route_xy @ rotation.T + translation,
        "gnss_in_local": (reference_xy[valid] - translation) @ rotation,
        "rotation": rotation,
        "translation": translation,
    }


def plot_height_map(
    path: str | Path,
    points: np.ndarray,
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    height_range: tuple[float, float],
    plot_stride: int,
    routes: list[dict] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = points[::plot_stride]

    figure, axis = plt.subplots(figsize=(11, 9))
    artist = axis.scatter(
        display[:, 0],
        display[:, 1],
        c=display[:, 2],
        cmap="viridis",
        vmin=height_range[0],
        vmax=height_range[1],
        s=0.35,
        linewidths=0,
        rasterized=True,
    )
    for route in routes or []:
        xy = route["xy"]
        axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=route["color"],
            linestyle=route.get("linestyle", "-"),
            linewidth=route.get("linewidth", 1.5),
            label=route["label"],
            zorder=3,
        )

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Map X [m]")
    axis.set_ylabel("Map Y [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if routes:
        axis.legend(loc="best")
    figure.colorbar(artist, ax=axis, label="Height Z [m]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_map_overlay(
    path: str | Path,
    kiss_points: np.ndarray,
    slam_points: np.ndarray,
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    plot_stride: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.scatter(
        kiss_points[::plot_stride, 0],
        kiss_points[::plot_stride, 1],
        s=0.4,
        color="#1f77b4",
        alpha=0.45,
        linewidths=0,
        label="KISS map",
        rasterized=True,
    )
    axis.scatter(
        slam_points[::plot_stride, 0],
        slam_points[::plot_stride, 1],
        s=0.4,
        color="#d62728",
        alpha=0.45,
        linewidths=0,
        label="SLAM map",
        rasterized=True,
    )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Map X [m]")
    axis.set_ylabel("Map Y [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_routes(
    path: str | Path,
    title: str,
    routes: list[dict],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    for route in routes:
        xy = route["xy"]
        axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=route["color"],
            linestyle=route.get("linestyle", "-"),
            linewidth=route.get("linewidth", 1.5),
            label=route["label"],
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("East [m]")
    axis.set_ylabel("North [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_plot_summary(path: str | Path, summary: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def generate_map_plots(
    kiss_map_path: str | Path,
    slam_map_path: str | Path,
    kiss_trajectory_path: str | Path,
    slam_trajectory_path: str | Path,
    gnss_csv_path: str | Path,
    gnss_time_offset_s: float,
    keyframes_path: str | Path,
    level_reference_trajectory_path: str | Path,
    source_id: int,
    target_id: int,
    output_directory: str | Path,
    loop_radius_m: float,
    plot_stride: int,
    minimum_z_m: float | None = None,
    maximum_z_m: float | None = None,
    kiss_poses: list[dict] | None = None,
    slam_poses: list[dict] | None = None,
    keyframes: list[dict] | None = None,
    level_reference_poses: list[dict] | None = None,
    selected_outputs: list[str] | tuple[str, ...] | None = None,
    map_plot_summary_path: str | Path | None = None,
) -> dict:
    if plot_stride < 1:
        raise ValueError("Plot stride must be at least one")
    if loop_radius_m <= 0.0:
        raise ValueError("Loop radius must be greater than zero")
    if minimum_z_m is not None and maximum_z_m is not None and minimum_z_m > maximum_z_m:
        raise ValueError("z-min must not be greater than z-max")

    kiss_points_all = load_ascii_pcd(kiss_map_path)
    slam_points_all = load_ascii_pcd(slam_map_path)
    kiss_points = filter_height(kiss_points_all, minimum_z_m, maximum_z_m)
    slam_points = filter_height(slam_points_all, minimum_z_m, maximum_z_m)
    if len(kiss_points) == 0 or len(slam_points) == 0:
        raise ValueError("Height filtering removed every point from at least one map")

    reference_timestamps, reference_xyz = (
        trajectory_arrays_from_poses(level_reference_poses)
        if level_reference_poses is not None
        else trajectory_arrays(level_reference_trajectory_path)
    )
    level_matrix, level_information = level_rotation(reference_xyz)
    kiss_timestamps, kiss_xyz = (
        trajectory_arrays_from_poses(kiss_poses)
        if kiss_poses is not None
        else trajectory_arrays(kiss_trajectory_path)
    )
    slam_timestamps, slam_xyz = (
        trajectory_arrays_from_poses(slam_poses)
        if slam_poses is not None
        else trajectory_arrays(slam_trajectory_path)
    )
    if len(kiss_timestamps) != len(slam_timestamps) or not np.allclose(
        kiss_timestamps, slam_timestamps, rtol=0.0, atol=1e-6
    ):
        raise ValueError("KISS and SLAM trajectory timestamps do not match")
    if len(reference_timestamps) != len(kiss_timestamps) or not np.allclose(
        reference_timestamps, kiss_timestamps, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Level-reference and plotted trajectory timestamps do not match")

    kiss_leveled = apply_rotation(kiss_xyz, level_matrix)
    slam_leveled = apply_rotation(slam_xyz, level_matrix)
    loop_centre, source_position, target_position = (
        calculate_leveled_loop_centre_from_keyframes(
            keyframes, source_id, target_id, level_matrix
        )
        if keyframes is not None
        else calculate_leveled_loop_centre(
            keyframes_path, source_id, target_id, level_matrix
        )
    )
    kiss_loop = crop_xy(kiss_points, loop_centre[:2], loop_radius_m)
    slam_loop = crop_xy(slam_points, loop_centre[:2], loop_radius_m)
    if len(kiss_loop) == 0 or len(slam_loop) == 0:
        raise ValueError("The loop crop contains no points from at least one map")

    gnss_raw = load_gnss_ground_truth(gnss_csv_path)
    gnss = gnss_to_enu(gnss_raw, origin_from_gnss(gnss_raw))
    reference = interpolate_ground_truth(
        gnss, kiss_timestamps, time_offset_s=gnss_time_offset_s
    )
    valid = reference["valid"].astype(bool, copy=False)
    if valid.sum() < 2:
        raise ValueError("Trajectory timestamps do not overlap corrected GNSS")
    gnss_xy = np.column_stack((reference["east_m"], reference["north_m"]))
    kiss_alignment = align_route_to_gnss(kiss_leveled[:, :2], gnss_xy, valid)
    slam_alignment = align_route_to_gnss(slam_leveled[:, :2], gnss_xy, valid)

    full_x, full_y = shared_xy_limits(kiss_points, slam_points)
    full_height = shared_height_range(
        kiss_points, slam_points, minimum_z_m, maximum_z_m
    )
    loop_x = (float(loop_centre[0] - loop_radius_m), float(loop_centre[0] + loop_radius_m))
    loop_y = (float(loop_centre[1] - loop_radius_m), float(loop_centre[1] + loop_radius_m))
    loop_height = shared_height_range(
        kiss_loop, slam_loop, minimum_z_m, maximum_z_m
    )

    output_path = Path(output_directory)
    suffix = f"{source_id}_{target_id}"
    outputs = {
        "kiss_map_topdown_height": output_path / "kiss_map_topdown_height.png",
        "slam_map_topdown_height": output_path / "slam_map_topdown_height.png",
        "kiss_map_loop_height": output_path / f"kiss_map_loop_{suffix}_height.png",
        "slam_map_loop_height": output_path / f"slam_map_loop_{suffix}_height.png",
        "map_overlay_full": output_path / "map_overlay_full.png",
        "map_overlay_loop": output_path / f"map_overlay_loop_{suffix}.png",
        "routes_kiss_slam_gnss": output_path / "routes_kiss_slam_gnss.png",
        "route_kiss_vs_gnss": output_path / "route_kiss_vs_gnss.png",
        "route_slam_vs_gnss": output_path / "route_slam_vs_gnss.png",
        "kiss_map_with_route_and_gnss": output_path / "kiss_map_with_route_and_gnss.png",
        "slam_map_with_route_and_gnss": output_path / "slam_map_with_route_and_gnss.png",
    }
    selected_names = set(outputs) if selected_outputs is None else set(selected_outputs)
    unknown_names = selected_names - set(outputs)
    if unknown_names:
        raise ValueError("Unknown map plot output: " + ", ".join(sorted(unknown_names)))
    if not selected_names:
        raise ValueError("At least one map plot output must be selected")
    generated_outputs: dict[str, Path] = {}

    def wants(name: str) -> bool:
        return name in selected_names

    if wants("kiss_map_topdown_height"):
        plot_height_map(
            outputs["kiss_map_topdown_height"], kiss_points, "KISS map height",
            full_x, full_y, full_height, plot_stride,
        )
        generated_outputs["kiss_map_topdown_height"] = outputs["kiss_map_topdown_height"]
    if wants("slam_map_topdown_height"):
        plot_height_map(
            outputs["slam_map_topdown_height"], slam_points, "SLAM map height",
            full_x, full_y, full_height, plot_stride,
        )
        generated_outputs["slam_map_topdown_height"] = outputs["slam_map_topdown_height"]
    if wants("kiss_map_loop_height"):
        plot_height_map(
            outputs["kiss_map_loop_height"], kiss_loop,
            f"KISS map height near loop {source_id} - {target_id}",
            loop_x, loop_y, loop_height, max(1, plot_stride // 2),
        )
        generated_outputs["kiss_map_loop_height"] = outputs["kiss_map_loop_height"]
    if wants("slam_map_loop_height"):
        plot_height_map(
            outputs["slam_map_loop_height"], slam_loop,
            f"SLAM map height near loop {source_id} - {target_id}",
            loop_x, loop_y, loop_height, max(1, plot_stride // 2),
        )
        generated_outputs["slam_map_loop_height"] = outputs["slam_map_loop_height"]
    if wants("map_overlay_full"):
        plot_map_overlay(
            outputs["map_overlay_full"], kiss_points, slam_points,
            "KISS and SLAM map overlay", full_x, full_y, plot_stride,
        )
        generated_outputs["map_overlay_full"] = outputs["map_overlay_full"]
    if wants("map_overlay_loop"):
        plot_map_overlay(
            outputs["map_overlay_loop"], kiss_loop, slam_loop,
            f"KISS and SLAM map overlay near loop {source_id} - {target_id}",
            loop_x, loop_y, max(1, plot_stride // 2),
        )
        generated_outputs["map_overlay_loop"] = outputs["map_overlay_loop"]

    gnss_valid = gnss_xy[valid]
    gnss_route = {
        "xy": gnss_valid, "label": "Corrected GNSS", "color": "#111111",
        "linestyle": "--", "linewidth": 2.0,
    }
    kiss_aligned_route = {
        "xy": kiss_alignment["aligned_valid"], "label": "KISS", "color": "#1f77b4",
    }
    slam_aligned_route = {
        "xy": slam_alignment["aligned_valid"], "label": "SLAM", "color": "#d62728",
    }
    if wants("routes_kiss_slam_gnss"):
        plot_routes(
            outputs["routes_kiss_slam_gnss"], "KISS, SLAM, and corrected GNSS routes",
            [gnss_route, kiss_aligned_route, slam_aligned_route],
        )
        generated_outputs["routes_kiss_slam_gnss"] = outputs["routes_kiss_slam_gnss"]
    if wants("route_kiss_vs_gnss"):
        plot_routes(
            outputs["route_kiss_vs_gnss"], "KISS route compared with corrected GNSS",
            [gnss_route, kiss_aligned_route],
        )
        generated_outputs["route_kiss_vs_gnss"] = outputs["route_kiss_vs_gnss"]
    if wants("route_slam_vs_gnss"):
        plot_routes(
            outputs["route_slam_vs_gnss"], "SLAM route compared with corrected GNSS",
            [gnss_route, slam_aligned_route],
        )
        generated_outputs["route_slam_vs_gnss"] = outputs["route_slam_vs_gnss"]

    kiss_local_routes = [
        {"xy": kiss_leveled[:, :2], "label": "KISS route", "color": "#d62728"},
        {
            "xy": kiss_alignment["gnss_in_local"], "label": "Corrected GNSS",
            "color": "#111111", "linestyle": "--", "linewidth": 2.0,
        },
    ]
    slam_local_routes = [
        {"xy": slam_leveled[:, :2], "label": "SLAM route", "color": "#d62728"},
        {
            "xy": slam_alignment["gnss_in_local"], "label": "Corrected GNSS",
            "color": "#111111", "linestyle": "--", "linewidth": 2.0,
        },
    ]
    if wants("kiss_map_with_route_and_gnss"):
        plot_height_map(
            outputs["kiss_map_with_route_and_gnss"], kiss_points,
            "KISS map with route and corrected GNSS", full_x, full_y,
            full_height, plot_stride, routes=kiss_local_routes,
        )
        generated_outputs["kiss_map_with_route_and_gnss"] = outputs["kiss_map_with_route_and_gnss"]
    if wants("slam_map_with_route_and_gnss"):
        plot_height_map(
            outputs["slam_map_with_route_and_gnss"], slam_points,
            "SLAM map with route and corrected GNSS", full_x, full_y,
            full_height, plot_stride, routes=slam_local_routes,
        )
        generated_outputs["slam_map_with_route_and_gnss"] = outputs["slam_map_with_route_and_gnss"]

    summary_path = (
        Path(map_plot_summary_path).expanduser()
        if map_plot_summary_path is not None
        else output_path / "map_plot_summary.json"
    )
    summary = {
        "inputs": {
            "kiss_map": str(kiss_map_path),
            "slam_map": str(slam_map_path),
            "kiss_trajectory": str(kiss_trajectory_path),
            "slam_trajectory": str(slam_trajectory_path),
            "gnss_csv": str(gnss_csv_path),
            "keyframes": str(keyframes_path),
            "level_reference_trajectory": str(level_reference_trajectory_path),
        },
        "gnss_time_offset_s": float(gnss_time_offset_s),
        "source_keyframe_id": int(source_id),
        "target_keyframe_id": int(target_id),
        "kiss_point_count_total": int(len(kiss_points_all)),
        "slam_point_count_total": int(len(slam_points_all)),
        "kiss_point_count_after_height_filter": int(len(kiss_points)),
        "slam_point_count_after_height_filter": int(len(slam_points)),
        "kiss_loop_point_count": int(len(kiss_loop)),
        "slam_loop_point_count": int(len(slam_loop)),
        "leveled_source_xyz_m": source_position.tolist(),
        "leveled_target_xyz_m": target_position.tolist(),
        "loop_centre_xyz_m": loop_centre.tolist(),
        "loop_radius_m": float(loop_radius_m),
        "plot_stride": int(plot_stride),
        "z_filter_m": {"minimum": minimum_z_m, "maximum": maximum_z_m},
        "full_map_shared_xy_limits_m": {"x": list(full_x), "y": list(full_y)},
        "loop_shared_xy_limits_m": {"x": list(loop_x), "y": list(loop_y)},
        "full_map_shared_height_range_m": list(full_height),
        "loop_region_shared_height_range_m": list(loop_height),
        "gnss_match_count": int(valid.sum()),
        "leveling": level_information,
        "outputs": {name: str(path) for name, path in generated_outputs.items()},
        "summary_path": str(summary_path),
    }
    write_plot_summary(summary_path, summary)
    return summary
