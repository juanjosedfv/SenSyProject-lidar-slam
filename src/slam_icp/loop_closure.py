from __future__ import annotations

import csv
import math
import sqlite3
import struct
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from baseline_icp.odometry import register_icp
from slam_icp.geometry import (
    invert_transform,
    pose_to_transform as canonical_pose_to_transform,
    transform_points as canonical_transform_points,
    yaw_from_transform,
)
from slam_icp.io import write_json
from preprocess import voxel_downsample
from slam_icp.pose_graph import wrap_angle

def find_bag_database(
    bag_path: str | Path,
) -> Path:
    path = Path(bag_path).expanduser()

    if path.is_file() and path.suffix == ".db3":
        return path

    if not path.exists():
        raise FileNotFoundError(
            f"ROS bag path does not exist: {path}"
        )

    database_files = sorted(
        path.glob("*.db3")
    )

    if not database_files:
        raise FileNotFoundError(
            f"No .db3 database found inside: {path}"
        )

    if len(database_files) > 1:
        raise ValueError(
            "This first implementation expects one "
            f".db3 file, but found {len(database_files)}"
        )

    return database_files[0]


def load_keyframe_by_id(
    path: str | Path,
    keyframe_id: int,
) -> dict:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Keyframe CSV does not exist: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["keyframe_id"]) != keyframe_id:
                continue

            return {
                "keyframe_id": int(
                    row["keyframe_id"]
                ),
                "timestamp": float(
                    row["timestamp"]
                ),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "qw": float(row["qw"]),
            }

    raise ValueError(
        f"Keyframe ID {keyframe_id} was not found "
        f"in {input_path}"
    )


def message_timestamp_seconds(
    message: PointCloud2,
) -> float:
    return (
        float(message.header.stamp.sec)
        + float(message.header.stamp.nanosec)
        * 1e-9
    )


def load_nearest_pointcloud(
    bag_path: str | Path,
    topic_name: str,
    target_timestamp: float,
    maximum_time_difference_s: float,
) -> tuple[PointCloud2, float]:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    database_path = find_bag_database(
        bag_path
    )

    best_message = None
    best_time_difference = math.inf

    with sqlite3.connect(
        database_path
    ) as connection:
        topic_row = connection.execute(
            """
            SELECT id
            FROM topics
            WHERE name = ?
            """,
            (topic_name,),
        ).fetchone()

        if topic_row is None:
            raise ValueError(
                f"Topic {topic_name} was not found "
                f"in {database_path}"
            )

        topic_id = int(topic_row[0])

        target_timestamp_ns = int(round(target_timestamp * 1e9))
        indexed_window_ns = max(
            int(math.ceil(maximum_time_difference_s * 1e9)) + 50_000_000,
            1_000_000_000,
        )
        indexed_rows = connection.execute(
            """
            SELECT data
            FROM messages
            WHERE topic_id = ?
              AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?)
            """,
            (
                topic_id,
                target_timestamp_ns - indexed_window_ns,
                target_timestamp_ns + indexed_window_ns,
                target_timestamp_ns,
            ),
        )

        for (raw_data,) in indexed_rows:
            message = deserialize_message(
                bytes(raw_data),
                PointCloud2,
            )
            time_difference = abs(
                message_timestamp_seconds(message) - target_timestamp
            )
            if time_difference < best_time_difference:
                best_message = message
                best_time_difference = time_difference

        if best_message is not None and best_time_difference <= maximum_time_difference_s:
            return best_message, best_time_difference

        best_message = None
        best_time_difference = math.inf
        rows = connection.execute(
            """
            SELECT data
            FROM messages
            WHERE topic_id = ?
            ORDER BY timestamp
            """,
            (topic_id,),
        )

        for (raw_data,) in rows:
            message = deserialize_message(
                bytes(raw_data),
                PointCloud2,
            )

            message_time = (
                message_timestamp_seconds(
                    message
                )
            )

            time_difference = abs(
                message_time
                - target_timestamp
            )

            if (
                time_difference
                < best_time_difference
            ):
                best_message = message
                best_time_difference = (
                    time_difference
                )

    if best_message is None:
        raise ValueError(
            f"No PointCloud2 messages found on "
            f"{topic_name}"
        )

    if (
        best_time_difference
        > maximum_time_difference_s
    ):
        raise ValueError(
            "Nearest LiDAR scan is too far from "
            "the keyframe timestamp: "
            f"{best_time_difference:.6f} s"
        )

    return (
        best_message,
        best_time_difference,
    )


def pointcloud2_to_xyz(
    message: PointCloud2,
) -> np.ndarray:
    fields = {
        field.name: field
        for field in message.fields
    }

    for required_name in ("x", "y", "z"):
        if required_name not in fields:
            raise ValueError(
                f"Point cloud is missing field: "
                f"{required_name}"
            )

    x_offset = fields["x"].offset
    y_offset = fields["y"].offset
    z_offset = fields["z"].offset

    endian_character = (
        ">"
        if message.is_bigendian
        else "<"
    )

    float_format = (
        endian_character + "f"
    )

    point_count = (
        message.width
        * message.height
    )

    points: list[tuple[float, float, float]] = []

    for point_index in range(point_count):
        base_offset = (
            point_index
            * message.point_step
        )

        try:
            x = struct.unpack_from(
                float_format,
                message.data,
                base_offset + x_offset,
            )[0]

            y = struct.unpack_from(
                float_format,
                message.data,
                base_offset + y_offset,
            )[0]

            z = struct.unpack_from(
                float_format,
                message.data,
                base_offset + z_offset,
            )[0]
        except struct.error:
            continue

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
        ):
            continue

        points.append((x, y, z))

    return np.asarray(
        points,
        dtype=np.float64,
    )


def filter_points_by_range(
    points: np.ndarray,
    minimum_range_m: float,
    maximum_range_m: float,
) -> np.ndarray:
    ranges = np.linalg.norm(
        points,
        axis=1,
    )

    valid = (
        (ranges >= minimum_range_m)
        & (ranges <= maximum_range_m)
    )

    return points[valid]


def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    norm = math.sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    )

    if norm == 0.0:
        raise ValueError(
            "Quaternion has zero length"
        )

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray,
    pose: dict,
) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(
        pose["qx"],
        pose["qy"],
        pose["qz"],
        pose["qw"],
    )

    translation = np.array(
        [
            pose["x"],
            pose["y"],
            pose["z"],
        ],
        dtype=np.float64,
    )

    return (
        points @ rotation.T
        + translation
    )


def crop_around_location(
    points: np.ndarray,
    centre_x: float,
    centre_y: float,
    radius_m: float,
) -> np.ndarray:
    dx = points[:, 0] - centre_x
    dy = points[:, 1] - centre_y

    distance = np.sqrt(
        dx * dx
        + dy * dy
    )

    return points[
        distance <= radius_m
    ]


def plot_loop_candidate(
    path: str | Path,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_keyframe: dict,
    target_keyframe: dict,
    plot_stride: int = 10,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_display = source_points[
        ::plot_stride
    ]

    target_display = target_points[
        ::plot_stride
    ]

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.scatter(
        source_display[:, 0],
        source_display[:, 1],
        s=1,
        alpha=0.5,
        label=(
            f"Keyframe "
            f"{source_keyframe['keyframe_id']}"
        ),
    )

    axis.scatter(
        target_display[:, 0],
        target_display[:, 1],
        s=1,
        alpha=0.5,
        label=(
            f"Keyframe "
            f"{target_keyframe['keyframe_id']}"
        ),
    )

    axis.scatter(
        [source_keyframe["x"]],
        [source_keyframe["y"]],
        s=80,
        marker="o",
        label="Source sensor position",
    )

    axis.scatter(
        [target_keyframe["x"]],
        [target_keyframe["y"]],
        s=80,
        marker="x",
        label="Target sensor position",
    )

    axis.set_title(
        "Loop-candidate LiDAR scan comparison"
    )

    axis.set_xlabel(
        "Local X [m]"
    )

    axis.set_ylabel(
        "Local Y [m]"
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

def pointcloud2_to_xyz_for_registration(message):
    return pointcloud2_to_xyz(message)



# Loop-candidate search helpers consolidated from the development module.

def load_pose_graph_nodes(
    path: str | Path,
) -> list[dict]:
    """
    Reads: pose_graph_nodes.csv
    It loads the position, timestamp, yaw and ID of every graph node
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pose-graph node file does not exist: {input_path}"
        )

    nodes: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Node CSV has no header: {input_path}"
            )

        missing_columns = (
            REQUIRED_NODE_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Node CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                node = {
                    "keyframe_id": int(
                        row["keyframe_id"]
                    ),
                    "timestamp": float(
                        row["timestamp"]
                    ),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw_rad": float(
                        row["yaw_rad"]
                    ),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in node CSV row "
                    f"{row_number}"
                ) from error

            nodes.append(node)

    if len(nodes) < 2:
        raise ValueError(
            "At least two graph nodes are required"
        )

    return nodes


def horizontal_distance(
    first_node: dict,
    second_node: dict,
) -> float:
    dx = second_node["x"] - first_node["x"]
    dy = second_node["y"] - first_node["y"]

    return math.sqrt(
        dx * dx
        + dy * dy
    )


def yaw_difference_degrees(
    first_node: dict,
    second_node: dict,
) -> float:
    difference_rad = wrap_angle(
        second_node["yaw_rad"]
        - first_node["yaw_rad"]
    )

    return abs(
        math.degrees(difference_rad)
    )


def find_loop_candidates(
    nodes: list[dict],
    search_radius_m: float,
    minimum_time_separation_s: float,
    minimum_id_separation: int,
    maximum_candidates: int | None = None,
) -> list[dict]:
    if search_radius_m <= 0.0:
        raise ValueError(
            "Search radius must be greater than zero"
        )

    if minimum_time_separation_s < 0.0:
        raise ValueError(
            "Minimum time separation cannot be negative"
        )

    if minimum_id_separation < 1:
        raise ValueError(
            "Minimum ID separation must be at least one"
        )

    candidates: list[dict] = []

    for source_index, source_node in enumerate(
        nodes[:-1]
    ):
        for target_node in nodes[
            source_index + 1:
        ]:
            id_separation = (
                target_node["keyframe_id"]
                - source_node["keyframe_id"]
            )

            if id_separation < minimum_id_separation:
                continue

            time_separation_s = (
                target_node["timestamp"]
                - source_node["timestamp"]
            )

            if (
                time_separation_s
                < minimum_time_separation_s
            ):
                continue

            distance_m = horizontal_distance(
                source_node,
                target_node,
            )

            if distance_m > search_radius_m:
                continue

            yaw_difference_deg = (
                yaw_difference_degrees(
                    source_node,
                    target_node,
                )
            )

            candidates.append(
                {
                    "source_id": (
                        source_node["keyframe_id"]
                    ),
                    "target_id": (
                        target_node["keyframe_id"]
                    ),
                    "source_timestamp": (
                        source_node["timestamp"]
                    ),
                    "target_timestamp": (
                        target_node["timestamp"]
                    ),
                    "source_x": source_node["x"],
                    "source_y": source_node["y"],
                    "target_x": target_node["x"],
                    "target_y": target_node["y"],
                    "distance_m": distance_m,
                    "time_separation_s": (
                        time_separation_s
                    ),
                    "id_separation": (
                        id_separation
                    ),
                    "yaw_difference_deg": (
                        yaw_difference_deg
                    ),
                }
            )

    candidates.sort(
        key=lambda candidate: (
            candidate["distance_m"],
            -candidate["time_separation_s"],
        )
    )

    if maximum_candidates is not None:
        candidates = candidates[
            :maximum_candidates
        ]

    for candidate_id, candidate in enumerate(
        candidates
    ):
        candidate["candidate_id"] = candidate_id

    return candidates


def write_loop_candidates_csv(
    path: str | Path,
    candidates: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "candidate_id",
        "source_id",
        "target_id",
        "distance_m",
        "yaw_difference_deg",
        "time_separation_s",
        "id_separation",
        "source_timestamp",
        "target_timestamp",
        "source_x",
        "source_y",
        "target_x",
        "target_y",
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

        for candidate in candidates:
            writer.writerow(
                {
                    field: candidate[field]
                    for field in fieldnames
                }
            )


def plot_loop_candidates(
    path: str | Path,
    nodes: list[dict],
    candidates: list[dict],
    number_to_plot: int = 15,
    trajectory_label: str = "KISS keyframe trajectory",
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_values = [
        node["x"]
        for node in nodes
    ]

    y_values = [
        node["y"]
        for node in nodes
    ]

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    axis.plot(
        x_values,
        y_values,
        linewidth=1.0,
        label=trajectory_label,
    )

    axis.scatter(
        x_values,
        y_values,
        s=7,
        label="Pose-graph nodes",
    )

    plotted_candidates = candidates[
        :number_to_plot
    ]

    for candidate in plotted_candidates:
        source_x = candidate["source_x"]
        source_y = candidate["source_y"]
        target_x = candidate["target_x"]
        target_y = candidate["target_y"]

        axis.plot(
            [source_x, target_x],
            [source_y, target_y],
            linewidth=1.0,
            linestyle="--",
        )

        axis.scatter(
            [source_x, target_x],
            [source_y, target_y],
            s=35,
        )

        middle_x = (
            source_x + target_x
        ) / 2.0

        middle_y = (
            source_y + target_y
        ) / 2.0

        axis.annotate(
            (
                f"C{candidate['candidate_id']}: "
                f"{candidate['source_id']}"
                f"↔{candidate['target_id']}"
            ),
            (middle_x, middle_y),
            fontsize=7,
        )

    axis.scatter(
        [nodes[0]["x"]],
        [nodes[0]["y"]],
        s=80,
        marker="o",
        label="Start",
    )

    axis.scatter(
        [nodes[-1]["x"]],
        [nodes[-1]["y"]],
        s=80,
        marker="x",
        label="End",
    )

    axis.set_title(
        "Possible SLAM loop-closure candidates"
    )

    axis.set_xlabel(
        "Local X [m]"
    )

    axis.set_ylabel(
        "Local Y [m]"
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


# Point-cloud registration helpers consolidated from the development module.

def register_scan_pair(
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    initial_transform: np.ndarray,
    maximum_correspondence_distance_m: float,
    maximum_iterations: int,
) -> dict:
    return register_icp(
        source=moving_points,
        target=reference_points,
        initial_transform=initial_transform,
        cfg={
            "icp_max_correspondence_distance_m": maximum_correspondence_distance_m,
            "icp_max_iterations": maximum_iterations,
            "icp_tolerance_m": 1e-4,
            "icp_max_correction_m": 0.0,
            "icp_degenerate_rmse_m": 1e-6,
        },
    )


# Loop-constraint registration helpers consolidated from the development module.

def pose_to_transform(pose: dict) -> np.ndarray:
    return canonical_pose_to_transform(pose)


def plot_registration(
    path: str | Path,
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    title: str,
    reference_id: int,
    moving_id: int,
    plot_stride: int,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_display = reference_points[
        ::plot_stride
    ]

    moving_display = moving_points[
        ::plot_stride
    ]

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.scatter(
        reference_display[:, 0],
        reference_display[:, 1],
        s=2,
        alpha=0.5,
        label=f"Reference KF{reference_id}",
    )

    axis.scatter(
        moving_display[:, 0],
        moving_display[:, 1],
        s=2,
        alpha=0.5,
        label=f"Moving KF{moving_id}",
    )

    axis.set_title(title)
    axis.set_xlabel("KF reference X [m]")
    axis.set_ylabel("KF reference Y [m]")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def register_loop_pair(
    keyframes: list[dict],
    lidar_bag_path: str | Path,
    lidar_topic: str,
    source_id: int,
    target_id: int,
    maximum_time_difference_s: float,
    minimum_range_m: float,
    maximum_range_m: float,
    voxel_size_m: float,
    maximum_correspondence_distance_m: float,
    maximum_iterations: int,
    output_directory: str | Path | None = None,
    plot_stride: int = 2,
    generate_plots: bool = False,
    write_constraint: bool = False,
) -> dict:
    keyframes_by_id = {
        int(keyframe["keyframe_id"]): keyframe
        for keyframe in keyframes
    }
    if source_id not in keyframes_by_id or target_id not in keyframes_by_id:
        raise ValueError(f"Accepted loop keyframes were not found: {source_id} -> {target_id}")
    source_keyframe = keyframes_by_id[source_id]
    target_keyframe = keyframes_by_id[target_id]

    source_message, source_dt = (
        load_nearest_pointcloud(
            bag_path=lidar_bag_path,
            topic_name=lidar_topic,
            target_timestamp=(
                source_keyframe["timestamp"]
            ),
            maximum_time_difference_s=(
                maximum_time_difference_s
            ),
        )
    )

    target_message, target_dt = (
        load_nearest_pointcloud(
            bag_path=lidar_bag_path,
            topic_name=lidar_topic,
            target_timestamp=(
                target_keyframe["timestamp"]
            ),
            maximum_time_difference_s=(
                maximum_time_difference_s
            ),
        )
    )

    source_cloud = pointcloud2_to_xyz_for_registration(
        source_message
    )

    target_cloud = pointcloud2_to_xyz_for_registration(
        target_message
    )

    source_cloud = filter_points_by_range(
        source_cloud,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
    )

    target_cloud = filter_points_by_range(
        target_cloud,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
    )

    source_cloud = voxel_downsample(
        source_cloud,
        voxel_size_m,
    )

    target_cloud = voxel_downsample(
        target_cloud,
        voxel_size_m,
    )

    source_pose = pose_to_transform(
        source_keyframe
    )

    target_pose = pose_to_transform(
        target_keyframe
    )

    initial_relative_transform = (
        invert_transform(source_pose)
        @ target_pose
    )

    result = register_scan_pair(
        moving_points=target_cloud,
        reference_points=source_cloud,
        initial_transform=initial_relative_transform,
        maximum_correspondence_distance_m=maximum_correspondence_distance_m,
        maximum_iterations=maximum_iterations,
    )

    refined_relative_transform = np.asarray(
        result["transformation"],
        dtype=np.float64,
    )

    target_with_initial_pose = canonical_transform_points(
        target_cloud,
        initial_relative_transform,
    )

    target_with_refined_pose = canonical_transform_points(
        target_cloud,
        refined_relative_transform,
    )

    output_path = Path(output_directory) if output_directory is not None else None

    before_plot = (
        output_path
        / "plots"
        / (
            f"loop_registration_"
            f"{source_id}_{target_id}_before.png"
        )
    ) if output_path is not None else None

    after_plot = (
        output_path
        / "plots"
        / (
            f"loop_registration_"
            f"{source_id}_{target_id}_after.png"
        )
    ) if output_path is not None else None

    if generate_plots:
        if output_path is None:
            raise ValueError("An output directory is required for registration plots")
        plot_registration(
            path=before_plot,
            reference_points=source_cloud,
            moving_points=target_with_initial_pose,
            title="Loop registration before ICP refinement",
            reference_id=source_id,
            moving_id=target_id,
            plot_stride=plot_stride,
        )
        plot_registration(
            path=after_plot,
            reference_points=source_cloud,
            moving_points=target_with_refined_pose,
            title="Loop registration after ICP refinement",
            reference_id=source_id,
            moving_id=target_id,
            plot_stride=plot_stride,
        )

    initial_yaw = yaw_from_transform(
        initial_relative_transform
    )

    refined_yaw = yaw_from_transform(
        refined_relative_transform
    )

    correction_transform = (
        refined_relative_transform
        @ invert_transform(
            initial_relative_transform
        )
    )

    correction_yaw = yaw_from_transform(
        correction_transform
    )

    loop_constraint = {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": "loop",
        "dx": float(
            refined_relative_transform[0, 3]
        ),
        "dy": float(
            refined_relative_transform[1, 3]
        ),
        "dyaw_rad": float(refined_yaw),
        "dyaw_deg": float(
            math.degrees(refined_yaw)
        ),
        "initial_dx": float(
            initial_relative_transform[0, 3]
        ),
        "initial_dy": float(
            initial_relative_transform[1, 3]
        ),
        "initial_dyaw_deg": float(
            math.degrees(initial_yaw)
        ),
        "correction_dx": float(
            correction_transform[0, 3]
        ),
        "correction_dy": float(
            correction_transform[1, 3]
        ),
        "correction_dyaw_deg": float(
            math.degrees(correction_yaw)
        ),
        "fitness": float(
            result["fitness"]
        ),
        "inlier_rmse_m": float(
            result["inlier_rmse"]
        ),
        "degenerate": bool(
            result.get("degenerate", False)
        ),
        "source_scan_pose_dt_s": float(
            source_dt
        ),
        "target_scan_pose_dt_s": float(
            target_dt
        ),
        "source_points_after_voxel": int(
            len(source_cloud)
        ),
        "target_points_after_voxel": int(
            len(target_cloud)
        ),
        "voxel_size_m": voxel_size_m,
        "maximum_correspondence_distance_m": (
            maximum_correspondence_distance_m
        ),
    }

    if write_constraint:
        if output_path is None:
            raise ValueError("An output directory is required for a loop checkpoint")
        constraint_path = write_json(
            output_path / f"loop_constraint_{source_id}_{target_id}.json",
            loop_constraint,
        )
        loop_constraint["constraint_path"] = str(constraint_path)
    if generate_plots:
        loop_constraint["before_plot"] = str(before_plot)
        loop_constraint["after_plot"] = str(after_plot)

    return loop_constraint


def _finite_metric(value: object) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def _loop_decision_metrics(registration: dict) -> dict:
    correction_dx = _finite_metric(registration.get("correction_dx"))
    correction_dy = _finite_metric(registration.get("correction_dy"))
    correction_translation = (
        math.hypot(correction_dx, correction_dy)
        if correction_dx is not None and correction_dy is not None
        else None
    )
    return {
        "source_id": int(registration["source_id"]),
        "target_id": int(registration["target_id"]),
        "fitness": _finite_metric(registration.get("fitness")),
        "inlier_rmse_m": _finite_metric(registration.get("inlier_rmse_m")),
        "correction_translation_m": correction_translation,
        "correction_yaw_deg": _finite_metric(registration.get("correction_dyaw_deg")),
    }


def evaluate_loop_registration(
    registration: dict,
    *,
    minimum_fitness: float,
    maximum_inlier_rmse_m: float,
    maximum_translation_correction_m: float,
    maximum_yaw_correction_deg: float,
) -> dict:
    """Apply conservative, dataset-independent gates to one loop registration."""

    decision = _loop_decision_metrics(registration)
    reasons: list[str] = []
    if bool(registration.get("degenerate", False)):
        reasons.append("degenerate registration")
    if decision["fitness"] is None:
        reasons.append("fitness is not finite")
    elif decision["fitness"] < minimum_fitness:
        reasons.append("fitness below minimum")
    if decision["inlier_rmse_m"] is None:
        reasons.append("inlier RMSE is not finite")
    elif decision["inlier_rmse_m"] > maximum_inlier_rmse_m:
        reasons.append("inlier RMSE above maximum")
    if decision["correction_translation_m"] is None:
        reasons.append("translation correction is not finite")
    elif decision["correction_translation_m"] > maximum_translation_correction_m:
        reasons.append("translation correction above maximum")
    if decision["correction_yaw_deg"] is None:
        reasons.append("yaw correction is not finite")
    elif abs(decision["correction_yaw_deg"]) > maximum_yaw_correction_deg:
        reasons.append("yaw correction above maximum")
    decision["accepted"] = not reasons
    decision["rejection_reason"] = "; ".join(reasons) if reasons else None
    return decision


def select_loop_registrations(
    *,
    mode: str,
    candidates: list[dict],
    manual_pairs: list[dict],
    register_pair: Callable[[int, int], dict],
    minimum_fitness: float = 0.30,
    maximum_inlier_rmse_m: float = 0.50,
    maximum_translation_correction_m: float = 5.0,
    maximum_yaw_correction_deg: float = 20.0,
    maximum_accepted_loops: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Register loops for the selected mode and return constraints and decisions."""

    if mode == "disabled":
        return [], []
    if mode == "manual":
        registrations: list[dict] = []
        decisions: list[dict] = []
        for pair in manual_pairs:
            registration = register_pair(
                int(pair["source_id"]),
                int(pair["target_id"]),
            )
            decision = _loop_decision_metrics(registration)
            decision["accepted"] = True
            decision["rejection_reason"] = None
            registrations.append(registration)
            decisions.append(decision)
        return registrations, decisions
    if mode != "auto":
        raise ValueError(f"Unsupported loop-closure mode: {mode}")

    decisions = []
    valid_registrations: list[tuple[int, dict, dict]] = []
    for candidate in candidates:
        source_id = int(candidate["source_id"])
        target_id = int(candidate["target_id"])
        try:
            registration = register_pair(source_id, target_id)
        except Exception as error:
            decisions.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "fitness": None,
                    "inlier_rmse_m": None,
                    "correction_translation_m": None,
                    "correction_yaw_deg": None,
                    "accepted": False,
                    "rejection_reason": (
                        f"registration failed: {type(error).__name__}: {error}"
                    ),
                }
            )
            continue

        decision = evaluate_loop_registration(
            registration,
            minimum_fitness=minimum_fitness,
            maximum_inlier_rmse_m=maximum_inlier_rmse_m,
            maximum_translation_correction_m=maximum_translation_correction_m,
            maximum_yaw_correction_deg=maximum_yaw_correction_deg,
        )
        decision_index = len(decisions)
        decisions.append(decision)
        if decision["accepted"]:
            valid_registrations.append((decision_index, registration, decision))

    valid_registrations.sort(
        key=lambda item: (
            -item[2]["fitness"],
            item[2]["inlier_rmse_m"],
            item[2]["correction_translation_m"],
            abs(item[2]["correction_yaw_deg"]),
            item[0],
        )
    )
    selected = valid_registrations[:maximum_accepted_loops]
    selected_indices = {item[0] for item in selected}
    for decision_index, _, decision in valid_registrations:
        if decision_index not in selected_indices:
            decision["accepted"] = False
            decision["rejection_reason"] = "valid registration not selected"
    return [item[1] for item in selected], decisions


def register_loop_candidate(
    keyframes_path: str | Path,
    lidar_bag_path: str | Path,
    lidar_topic: str,
    source_id: int,
    target_id: int,
    maximum_time_difference_s: float,
    minimum_range_m: float,
    maximum_range_m: float,
    voxel_size_m: float,
    maximum_correspondence_distance_m: float,
    maximum_iterations: int,
    output_directory: str | Path,
    plot_stride: int,
) -> dict:
    source = load_keyframe_by_id(keyframes_path, source_id)
    target = load_keyframe_by_id(keyframes_path, target_id)
    return register_loop_pair(
        keyframes=[source, target],
        lidar_bag_path=lidar_bag_path,
        lidar_topic=lidar_topic,
        source_id=source_id,
        target_id=target_id,
        maximum_time_difference_s=maximum_time_difference_s,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
        voxel_size_m=voxel_size_m,
        maximum_correspondence_distance_m=maximum_correspondence_distance_m,
        maximum_iterations=maximum_iterations,
        output_directory=output_directory,
        plot_stride=plot_stride,
        generate_plots=True,
        write_constraint=True,
    )
