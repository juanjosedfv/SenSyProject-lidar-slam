from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from slam_icp.keyframes import (
    quaternion_to_yaw,
    wrapped_angle_difference,
)


REQUIRED_KEYFRAME_COLUMNS = {
    "keyframe_id",
    "pose_index",
    "timestamp",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "yaw_deg",
    "backend",
    "parent_frame",
    "child_frame",
}


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(
        math.sin(angle_rad),
        math.cos(angle_rad),
    )


def load_keyframes(path: str | Path) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Keyframe CSV does not exist: {input_path}"
        )

    keyframes: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Keyframe CSV has no header: {input_path}"
            )

        missing_columns = (
            REQUIRED_KEYFRAME_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Keyframe CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                keyframe = {
                    "keyframe_id": int(row["keyframe_id"]),
                    "pose_index": int(row["pose_index"]),
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                    "yaw_rad": math.radians(
                        float(row["yaw_deg"])
                    ),
                    "backend": row["backend"],
                    "parent_frame": row["parent_frame"],
                    "child_frame": row["child_frame"],
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in keyframe CSV row "
                    f"{row_number}"
                ) from error

            keyframes.append(keyframe)

    if len(keyframes) < 2:
        raise ValueError(
            "At least two keyframes are required "
            "to create a pose graph"
        )

    for expected_id, keyframe in enumerate(keyframes):
        if keyframe["keyframe_id"] != expected_id:
            raise ValueError(
                "Keyframe IDs must begin at zero "
                "and increase consecutively"
            )

    for previous, current in zip(
        keyframes[:-1],
        keyframes[1:],
    ):
        if current["timestamp"] <= previous["timestamp"]:
            raise ValueError(
                "Keyframe timestamps must be "
                "strictly increasing"
            )

    return keyframes


def calculate_relative_pose_2d(
    source: dict,
    target: dict,
) -> dict:
    world_dx = target["x"] - source["x"]
    world_dy = target["y"] - source["y"]

    source_yaw = source["yaw_rad"]

    cos_yaw = math.cos(source_yaw)
    sin_yaw = math.sin(source_yaw)

    local_dx = (
        cos_yaw * world_dx
        + sin_yaw * world_dy
    )

    local_dy = (
        -sin_yaw * world_dx
        + cos_yaw * world_dy
    )

    local_dyaw = wrap_angle(
        target["yaw_rad"]
        - source["yaw_rad"]
    )

    return {
        "dx": local_dx,
        "dy": local_dy,
        "dyaw_rad": local_dyaw,
    }


def build_odometry_edges(
    keyframes: list[dict],
) -> list[dict]:
    edges: list[dict] = []

    for source, target in zip(
        keyframes[:-1],
        keyframes[1:],
    ):
        relative_pose = calculate_relative_pose_2d(
            source,
            target,
        )

        translation_m = math.sqrt(
            relative_pose["dx"] ** 2
            + relative_pose["dy"] ** 2
        )

        edge = {
            "edge_id": len(edges),
            "source_id": source["keyframe_id"],
            "target_id": target["keyframe_id"],
            "edge_type": "odometry",
            "dx": relative_pose["dx"],
            "dy": relative_pose["dy"],
            "dyaw_rad": relative_pose["dyaw_rad"],
            "dyaw_deg": math.degrees(
                relative_pose["dyaw_rad"]
            ),
            "translation_m": translation_m,
            "time_difference_s": (
                target["timestamp"]
                - source["timestamp"]
            ),
        }

        edges.append(edge)

    return edges


def apply_relative_pose_2d(
    source_pose: dict,
    edge: dict,
) -> dict:
    source_yaw = source_pose["yaw_rad"]

    cos_yaw = math.cos(source_yaw)
    sin_yaw = math.sin(source_yaw)

    world_dx = (
        cos_yaw * edge["dx"]
        - sin_yaw * edge["dy"]
    )

    world_dy = (
        sin_yaw * edge["dx"]
        + cos_yaw * edge["dy"]
    )

    target_x = source_pose["x"] + world_dx
    target_y = source_pose["y"] + world_dy

    target_yaw = wrap_angle(
        source_yaw
        + edge["dyaw_rad"]
    )

    return {
        "x": target_x,
        "y": target_y,
        "yaw_rad": target_yaw,
    }


def reconstruct_graph_trajectory(
    keyframes: list[dict],
    edges: list[dict],
) -> list[dict]:
    first_keyframe = keyframes[0]

    reconstructed = [
        {
            "keyframe_id": first_keyframe["keyframe_id"],
            "timestamp": first_keyframe["timestamp"],
            "x": first_keyframe["x"],
            "y": first_keyframe["y"],
            "yaw_rad": first_keyframe["yaw_rad"],
        }
    ]

    for edge in edges:
        current_pose = reconstructed[-1]

        next_pose = apply_relative_pose_2d(
            current_pose,
            edge,
        )

        target_keyframe = keyframes[
            edge["target_id"]
        ]

        reconstructed.append(
            {
                "keyframe_id": edge["target_id"],
                "timestamp": target_keyframe["timestamp"],
                "x": next_pose["x"],
                "y": next_pose["y"],
                "yaw_rad": next_pose["yaw_rad"],
            }
        )

    return reconstructed


def calculate_reconstruction_errors(
    keyframes: list[dict],
    reconstructed: list[dict],
) -> dict:
    position_errors: list[float] = []
    yaw_errors_deg: list[float] = []

    for original, rebuilt in zip(
        keyframes,
        reconstructed,
    ):
        dx = rebuilt["x"] - original["x"]
        dy = rebuilt["y"] - original["y"]

        position_error = math.sqrt(
            dx * dx
            + dy * dy
        )

        yaw_error = wrap_angle(
            rebuilt["yaw_rad"]
            - original["yaw_rad"]
        )

        position_errors.append(position_error)

        yaw_errors_deg.append(
            abs(math.degrees(yaw_error))
        )

    return {
        "maximum_position_error_m": max(
            position_errors
        ),
        "mean_position_error_m": (
            sum(position_errors)
            / len(position_errors)
        ),
        "maximum_yaw_error_deg": max(
            yaw_errors_deg
        ),
        "mean_yaw_error_deg": (
            sum(yaw_errors_deg)
            / len(yaw_errors_deg)
        ),
    }


def write_nodes_csv(
    path: str | Path,
    keyframes: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "keyframe_id",
        "pose_index",
        "timestamp",
        "x",
        "y",
        "yaw_rad",
        "yaw_deg",
        "backend",
        "parent_frame",
        "child_frame",
        "fixed",
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

        for keyframe in keyframes:
            writer.writerow(
                {
                    "keyframe_id": (
                        keyframe["keyframe_id"]
                    ),
                    "pose_index": (
                        keyframe["pose_index"]
                    ),
                    "timestamp": (
                        keyframe["timestamp"]
                    ),
                    "x": keyframe["x"],
                    "y": keyframe["y"],
                    "yaw_rad": (
                        keyframe["yaw_rad"]
                    ),
                    "yaw_deg": math.degrees(
                        keyframe["yaw_rad"]
                    ),
                    "backend": (
                        keyframe["backend"]
                    ),
                    "parent_frame": (
                        keyframe["parent_frame"]
                    ),
                    "child_frame": (
                        keyframe["child_frame"]
                    ),
                    "fixed": (
                        1
                        if keyframe["keyframe_id"] == 0
                        else 0
                    ),
                }
            )


def write_edges_csv(
    path: str | Path,
    edges: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "edge_id",
        "source_id",
        "target_id",
        "edge_type",
        "dx",
        "dy",
        "dyaw_rad",
        "dyaw_deg",
        "translation_m",
        "time_difference_s",
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

        for edge in edges:
            writer.writerow(
                {
                    field: edge[field]
                    for field in fieldnames
                }
            )


def write_reconstructed_csv(
    path: str | Path,
    reconstructed: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "keyframe_id",
        "timestamp",
        "x",
        "y",
        "yaw_rad",
        "yaw_deg",
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

        for pose in reconstructed:
            writer.writerow(
                {
                    "keyframe_id": (
                        pose["keyframe_id"]
                    ),
                    "timestamp": (
                        pose["timestamp"]
                    ),
                    "x": pose["x"],
                    "y": pose["y"],
                    "yaw_rad": (
                        pose["yaw_rad"]
                    ),
                    "yaw_deg": math.degrees(
                        pose["yaw_rad"]
                    ),
                }
            )


def write_graph_summary(
    path: str | Path,
    keyframes: list[dict],
    edges: list[dict],
    reconstruction_errors: dict,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_distances = [
        edge["translation_m"]
        for edge in edges
    ]

    edge_yaw_changes = [
        abs(edge["dyaw_deg"])
        for edge in edges
    ]

    summary = {
        "backend": keyframes[0]["backend"],
        "node_count": len(keyframes),
        "odometry_edge_count": len(edges),
        "loop_edge_count": 0,
        "fixed_node_id": 0,
        "mean_odometry_edge_distance_m": (
            sum(edge_distances)
            / len(edge_distances)
        ),
        "maximum_odometry_edge_distance_m": max(
            edge_distances
        ),
        "mean_odometry_edge_yaw_deg": (
            sum(edge_yaw_changes)
            / len(edge_yaw_changes)
        ),
        "maximum_odometry_edge_yaw_deg": max(
            edge_yaw_changes
        ),
        **reconstruction_errors,
    }

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_initial_pose_graph(
    path: str | Path,
    keyframes: list[dict],
    label_every: int = 25,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_values = [
        keyframe["x"]
        for keyframe in keyframes
    ]

    y_values = [
        keyframe["y"]
        for keyframe in keyframes
    ]

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.plot(
        x_values,
        y_values,
        linewidth=1.0,
        label="Odometry edges",
    )

    axis.scatter(
        x_values,
        y_values,
        s=10,
        label="Pose-graph nodes",
    )

    axis.scatter(
        [keyframes[0]["x"]],
        [keyframes[0]["y"]],
        s=80,
        marker="o",
        label="Fixed first node",
    )

    axis.scatter(
        [keyframes[-1]["x"]],
        [keyframes[-1]["y"]],
        s=80,
        marker="x",
        label="Final node",
    )

    if label_every > 0:
        for keyframe in keyframes[::label_every]:
            axis.annotate(
                str(keyframe["keyframe_id"]),
                (
                    keyframe["x"],
                    keyframe["y"],
                ),
                fontsize=7,
            )

    axis.set_title(
        "Initial SLAM pose graph"
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

# Full-trajectory correction helpers consolidated from the development module.

REQUIRED_TRAJECTORY_COLUMNS = {
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
}


def load_full_trajectory(
    path: str | Path,
) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Trajectory CSV does not exist: {input_path}"
        )

    poses: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Trajectory CSV has no header: {input_path}"
            )

        missing_columns = (
            REQUIRED_TRAJECTORY_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Trajectory CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for pose_index, row in enumerate(reader):
            try:
                pose = {
                    "pose_index": pose_index,
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                    "backend": row["backend"],
                    "parent_frame": row["parent_frame"],
                    "child_frame": row["child_frame"],
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid trajectory value in CSV row "
                    f"{pose_index + 2}"
                ) from error

            poses.append(pose)

    if not poses:
        raise ValueError(
            "The full trajectory contains no poses"
        )

    for previous, current in zip(
        poses[:-1],
        poses[1:],
    ):
        if current["timestamp"] <= previous["timestamp"]:
            raise ValueError(
                "Trajectory timestamps must be "
                "strictly increasing"
            )

    return poses


def load_keyframe_pose_indices(
    path: str | Path,
) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Keyframe CSV does not exist: {input_path}"
        )

    keyframes: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                keyframes.append(
                    {
                        "keyframe_id": int(
                            row["keyframe_id"]
                        ),
                        "pose_index": int(
                            row["pose_index"]
                        ),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid keyframe value in CSV row "
                    f"{row_number}"
                ) from error

    if not keyframes:
        raise ValueError(
            "No keyframes were loaded"
        )

    return keyframes


def load_optimized_keyframe_corrections(
    optimized_nodes_path: str | Path,
    keyframe_indices: list[dict],
) -> list[dict]:
    input_path = Path(
        optimized_nodes_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Optimized-node CSV does not exist: "
            f"{input_path}"
        )

    optimized_by_id: dict[int, dict] = {}

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                keyframe_id = int(
                    row["keyframe_id"]
                )

                optimized_by_id[keyframe_id] = {
                    "x_before": float(
                        row["x_before"]
                    ),
                    "y_before": float(
                        row["y_before"]
                    ),
                    "yaw_before_rad": float(
                        row["yaw_before_rad"]
                    ),
                    "x_after": float(
                        row["x_after"]
                    ),
                    "y_after": float(
                        row["y_after"]
                    ),
                    "yaw_after_rad": float(
                        row["yaw_after_rad"]
                    ),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid optimized-node value "
                    f"in CSV row {row_number}"
                ) from error

    corrections: list[dict] = []

    for keyframe in keyframe_indices:
        keyframe_id = keyframe["keyframe_id"]

        if keyframe_id not in optimized_by_id:
            raise ValueError(
                f"Optimized pose is missing for "
                f"Keyframe {keyframe_id}"
            )

        optimized = optimized_by_id[
            keyframe_id
        ]

        correction_dyaw_rad = (
            wrapped_angle_difference(
                optimized["yaw_after_rad"],
                optimized["yaw_before_rad"],
            )
        )

        corrections.append(
            {
                "keyframe_id": keyframe_id,
                "pose_index": (
                    keyframe["pose_index"]
                ),
                "correction_dx": (
                    optimized["x_after"]
                    - optimized["x_before"]
                ),
                "correction_dy": (
                    optimized["y_after"]
                    - optimized["y_before"]
                ),
                "correction_dyaw_rad": (
                    correction_dyaw_rad
                ),
                "x_after": optimized["x_after"],
                "y_after": optimized["y_after"],
                "yaw_after_rad": (
                    optimized["yaw_after_rad"]
                ),
            }
        )

    corrections.sort(
        key=lambda correction: (
            correction["pose_index"]
        )
    )

    return corrections


def build_keyframe_corrections(
    keyframes: list[dict],
    original_nodes: list[dict],
    optimized_nodes: list[dict],
) -> list[dict]:
    """Create full-trajectory correction anchors without a CSV round trip."""

    if not (
        len(keyframes) == len(original_nodes) == len(optimized_nodes)
    ):
        raise ValueError("Keyframe and graph node counts must match")

    corrections: list[dict] = []
    for keyframe, original, optimized in zip(
        keyframes, original_nodes, optimized_nodes
    ):
        keyframe_id = int(keyframe["keyframe_id"])
        if keyframe_id != int(original["keyframe_id"]) or keyframe_id != int(
            optimized["keyframe_id"]
        ):
            raise ValueError("Keyframe and graph node IDs must match")
        corrections.append(
            {
                "keyframe_id": keyframe_id,
                "pose_index": int(keyframe["pose_index"]),
                "correction_dx": float(optimized["x"] - original["x"]),
                "correction_dy": float(optimized["y"] - original["y"]),
                "correction_dyaw_rad": wrapped_angle_difference(
                    optimized["yaw_rad"], original["yaw_rad"]
                ),
                "x_after": float(optimized["x"]),
                "y_after": float(optimized["y"]),
                "yaw_after_rad": float(optimized["yaw_rad"]),
            }
        )

    corrections.sort(key=lambda correction: correction["pose_index"])
    return corrections


def interpolate_angle(
    start_angle: float,
    end_angle: float,
    alpha: float,
) -> float:
    difference = wrapped_angle_difference(
        end_angle,
        start_angle,
    )

    interpolated = (
        start_angle
        + alpha * difference
    )

    return math.atan2(
        math.sin(interpolated),
        math.cos(interpolated),
    )


def interpolate_correction(
    first_correction: dict,
    second_correction: dict,
    pose_index: int,
) -> dict:
    first_index = first_correction[
        "pose_index"
    ]

    second_index = second_correction[
        "pose_index"
    ]

    index_span = second_index - first_index

    if index_span <= 0:
        raise ValueError(
            "Keyframe pose indices must increase"
        )

    alpha = (
        pose_index - first_index
    ) / index_span

    alpha = max(
        0.0,
        min(1.0, alpha),
    )

    correction_dx = (
        first_correction["correction_dx"]
        + alpha
        * (
            second_correction["correction_dx"]
            - first_correction["correction_dx"]
        )
    )

    correction_dy = (
        first_correction["correction_dy"]
        + alpha
        * (
            second_correction["correction_dy"]
            - first_correction["correction_dy"]
        )
    )

    correction_dyaw_rad = interpolate_angle(
        first_correction[
            "correction_dyaw_rad"
        ],
        second_correction[
            "correction_dyaw_rad"
        ],
        alpha,
    )

    return {
        "dx": correction_dx,
        "dy": correction_dy,
        "dyaw_rad": correction_dyaw_rad,
    }


def multiply_quaternions_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right

    x = (
        lw * rx
        + lx * rw
        + ly * rz
        - lz * ry
    )

    y = (
        lw * ry
        - lx * rz
        + ly * rw
        + lz * rx
    )

    z = (
        lw * rz
        + lx * ry
        - ly * rx
        + lz * rw
    )

    w = (
        lw * rw
        - lx * rx
        - ly * ry
        - lz * rz
    )

    norm = math.sqrt(
        x * x
        + y * y
        + z * z
        + w * w
    )

    if norm == 0.0:
        raise ValueError(
            "Quaternion multiplication produced "
            "a zero-length quaternion"
        )

    return (
        x / norm,
        y / norm,
        z / norm,
        w / norm,
    )


def apply_yaw_correction(
    pose: dict,
    correction_dyaw_rad: float,
) -> tuple[float, float, float, float]:
    half_angle = (
        correction_dyaw_rad / 2.0
    )

    yaw_correction_quaternion = (
        0.0,
        0.0,
        math.sin(half_angle),
        math.cos(half_angle),
    )

    original_quaternion = (
        pose["qx"],
        pose["qy"],
        pose["qz"],
        pose["qw"],
    )

    return multiply_quaternions_xyzw(
        yaw_correction_quaternion,
        original_quaternion,
    )


def apply_keyframe_corrections(
    poses: list[dict],
    corrections: list[dict],
) -> list[dict]:
    if corrections[0]["pose_index"] != 0:
        raise ValueError(
            "The first keyframe must correspond "
            "to the first trajectory pose"
        )

    if (
        corrections[-1]["pose_index"]
        != len(poses) - 1
    ):
        raise ValueError(
            "The final keyframe must correspond "
            "to the final trajectory pose"
        )

    corrected_poses: list[dict] = []

    segment_index = 0

    for pose in poses:
        pose_index = pose["pose_index"]

        while (
            segment_index
            < len(corrections) - 2
            and pose_index
            > corrections[
                segment_index + 1
            ]["pose_index"]
        ):
            segment_index += 1

        first_correction = corrections[
            segment_index
        ]

        second_correction = corrections[
            segment_index + 1
        ]

        correction = interpolate_correction(
            first_correction,
            second_correction,
            pose_index,
        )

        corrected_quaternion = (
            apply_yaw_correction(
                pose,
                correction["dyaw_rad"],
            )
        )

        corrected_pose = {
            **pose,
            "x": (
                pose["x"]
                + correction["dx"]
            ),
            "y": (
                pose["y"]
                + correction["dy"]
            ),
            "qx": corrected_quaternion[0],
            "qy": corrected_quaternion[1],
            "qz": corrected_quaternion[2],
            "qw": corrected_quaternion[3],
            "trajectory_type": "slam_optimized",
            "correction_dx": correction["dx"],
            "correction_dy": correction["dy"],
            "correction_dyaw_deg": (
                math.degrees(
                    correction["dyaw_rad"]
                )
            ),
        }

        corrected_poses.append(
            corrected_pose
        )

    return corrected_poses


def calculate_correction_summary(
    corrected_poses: list[dict],
    corrections: list[dict],
) -> dict:
    position_corrections = [
        math.sqrt(
            pose["correction_dx"] ** 2
            + pose["correction_dy"] ** 2
        )
        for pose in corrected_poses
    ]

    yaw_corrections_deg = [
        abs(pose["correction_dyaw_deg"])
        for pose in corrected_poses
    ]

    keyframe_position_errors: list[float] = []
    keyframe_yaw_errors_deg: list[float] = []

    for correction in corrections:
        corrected_pose = corrected_poses[
            correction["pose_index"]
        ]

        position_error = math.sqrt(
            (
                corrected_pose["x"]
                - correction["x_after"]
            ) ** 2
            + (
                corrected_pose["y"]
                - correction["y_after"]
            ) ** 2
        )

        corrected_yaw = quaternion_to_yaw(
            corrected_pose["qx"],
            corrected_pose["qy"],
            corrected_pose["qz"],
            corrected_pose["qw"],
        )

        yaw_error = (
            wrapped_angle_difference(
                corrected_yaw,
                correction["yaw_after_rad"],
            )
        )

        keyframe_position_errors.append(
            position_error
        )

        keyframe_yaw_errors_deg.append(
            abs(math.degrees(yaw_error))
        )

    return {
        "pose_count": len(corrected_poses),
        "keyframe_count": len(corrections),
        "mean_position_correction_m": float(
            np.mean(position_corrections)
        ),
        "maximum_position_correction_m": float(
            np.max(position_corrections)
        ),
        "mean_yaw_correction_deg": float(
            np.mean(yaw_corrections_deg)
        ),
        "maximum_yaw_correction_deg": float(
            np.max(yaw_corrections_deg)
        ),
        "maximum_keyframe_position_mismatch_m": float(
            np.max(keyframe_position_errors)
        ),
        "maximum_keyframe_yaw_mismatch_deg": float(
            np.max(keyframe_yaw_errors_deg)
        ),
        "first_timestamp": corrected_poses[0][
            "timestamp"
        ],
        "last_timestamp": corrected_poses[-1][
            "timestamp"
        ],
    }


def write_corrected_trajectory_csv(
    path: str | Path,
    corrected_poses: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "pose_index",
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
        "trajectory_type",
        "correction_dx",
        "correction_dy",
        "correction_dyaw_deg",
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

        for pose in corrected_poses:
            writer.writerow(
                {
                    field: pose[field]
                    for field in fieldnames
                }
            )


def write_correction_summary(
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


def plot_full_trajectory_correction(
    path: str | Path,
    original_poses: list[dict],
    corrected_poses: list[dict],
    original_label: str = "Original KISS trajectory",
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_x = [
        pose["x"]
        for pose in original_poses
    ]

    original_y = [
        pose["y"]
        for pose in original_poses
    ]

    corrected_x = [
        pose["x"]
        for pose in corrected_poses
    ]

    corrected_y = [
        pose["y"]
        for pose in corrected_poses
    ]

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    axis.plot(
        original_x,
        original_y,
        linewidth=1.0,
        label=original_label,
    )

    axis.plot(
        corrected_x,
        corrected_y,
        linewidth=1.5,
        label="SLAM-corrected trajectory",
    )

    axis.scatter(
        [corrected_poses[0]["x"]],
        [corrected_poses[0]["y"]],
        s=80,
        marker="o",
        label="Fixed start",
    )

    axis.scatter(
        [corrected_poses[-1]["x"]],
        [corrected_poses[-1]["y"]],
        s=80,
        marker="x",
        label="Corrected end",
    )

    axis.set_title(
        "Full trajectory before and after SLAM correction"
    )

    axis.set_xlabel("Local X [m]")
    axis.set_ylabel("Local Y [m]")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


# Pose-graph optimization helpers consolidated from the development module.

def load_graph_nodes(
    path: str | Path,
) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pose-graph node CSV does not exist: {input_path}"
        )

    nodes: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            try:
                node = {
                    "keyframe_id": int(row["keyframe_id"]),
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw_rad": float(row["yaw_rad"]),
                    "fixed": bool(int(row["fixed"])),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in node CSV row {row_number}"
                ) from error

            nodes.append(node)

    if len(nodes) < 2:
        raise ValueError(
            "At least two nodes are required for optimization"
        )

    for expected_id, node in enumerate(nodes):
        if node["keyframe_id"] != expected_id:
            raise ValueError(
                "Node IDs must start at zero and be consecutive"
            )

    if not nodes[0]["fixed"]:
        raise ValueError(
            "The first graph node must be fixed"
        )

    return nodes


def load_odometry_edges(
    path: str | Path,
) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pose-graph edge CSV does not exist: {input_path}"
        )

    edges: list[dict] = []

    with input_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            try:
                edge = {
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_type": row["edge_type"],
                    "dx": float(row["dx"]),
                    "dy": float(row["dy"]),
                    "dyaw_rad": float(row["dyaw_rad"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in edge CSV row {row_number}"
                ) from error

            edges.append(edge)

    if not edges:
        raise ValueError(
            "No odometry edges were loaded"
        )

    return edges


def load_loop_constraint(
    path: str | Path,
) -> dict:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Loop-constraint JSON does not exist: {input_path}"
        )

    data = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    required_fields = {
        "source_id",
        "target_id",
        "dx",
        "dy",
        "dyaw_rad",
    }

    missing_fields = (
        required_fields - set(data)
    )

    if missing_fields:
        raise ValueError(
            "Loop constraint is missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    return {
        "source_id": int(data["source_id"]),
        "target_id": int(data["target_id"]),
        "edge_type": "loop",
        "dx": float(data["dx"]),
        "dy": float(data["dy"]),
        "dyaw_rad": float(data["dyaw_rad"]),
        "fitness": float(data.get("fitness", math.nan)),
        "inlier_rmse_m": float(
            data.get("inlier_rmse_m", math.nan)
        ),
    }


def nodes_to_state(
    nodes: list[dict],
) -> np.ndarray:
    state_values: list[float] = []

    for node in nodes[1:]:
        state_values.extend(
            [
                node["x"],
                node["y"],
                node["yaw_rad"],
            ]
        )

    return np.asarray(
        state_values,
        dtype=np.float64,
    )


def state_to_poses(
    state: np.ndarray,
    fixed_node: dict,
) -> list[dict]:
    poses = [
        {
            "x": fixed_node["x"],
            "y": fixed_node["y"],
            "yaw_rad": fixed_node["yaw_rad"],
        }
    ]

    for start_index in range(
        0,
        len(state),
        3,
    ):
        poses.append(
            {
                "x": float(state[start_index]),
                "y": float(state[start_index + 1]),
                "yaw_rad": float(
                    state[start_index + 2]
                ),
            }
        )

    return poses


def calculate_relative_pose(
    source_pose: dict,
    target_pose: dict,
) -> dict:
    world_dx = (
        target_pose["x"]
        - source_pose["x"]
    )

    world_dy = (
        target_pose["y"]
        - source_pose["y"]
    )

    source_yaw = source_pose["yaw_rad"]

    cos_yaw = math.cos(source_yaw)
    sin_yaw = math.sin(source_yaw)

    local_dx = (
        cos_yaw * world_dx
        + sin_yaw * world_dy
    )

    local_dy = (
        -sin_yaw * world_dx
        + cos_yaw * world_dy
    )

    local_dyaw = wrap_angle(
        target_pose["yaw_rad"]
        - source_pose["yaw_rad"]
    )

    return {
        "dx": local_dx,
        "dy": local_dy,
        "dyaw_rad": local_dyaw,
    }


def calculate_edge_error(
    poses: list[dict],
    edge: dict,
) -> dict:
    predicted = calculate_relative_pose(
        poses[edge["source_id"]],
        poses[edge["target_id"]],
    )

    return {
        "dx": predicted["dx"] - edge["dx"],
        "dy": predicted["dy"] - edge["dy"],
        "dyaw_rad": wrap_angle(
            predicted["dyaw_rad"]
            - edge["dyaw_rad"]
        ),
    }


def build_residual_vector(
    state: np.ndarray,
    fixed_node: dict,
    edges: list[dict],
    odometry_translation_sigma_m: float,
    odometry_yaw_sigma_rad: float,
    loop_translation_sigma_m: float,
    loop_yaw_sigma_rad: float,
) -> np.ndarray:
    poses = state_to_poses(
        state,
        fixed_node,
    )

    residuals: list[float] = []

    for edge in edges:
        error = calculate_edge_error(
            poses,
            edge,
        )

        if edge["edge_type"] == "loop":
            translation_sigma = (
                loop_translation_sigma_m
            )
            yaw_sigma = loop_yaw_sigma_rad
        else:
            translation_sigma = (
                odometry_translation_sigma_m
            )
            yaw_sigma = odometry_yaw_sigma_rad

        residuals.extend(
            [
                error["dx"] / translation_sigma,
                error["dy"] / translation_sigma,
                error["dyaw_rad"] / yaw_sigma,
            ]
        )

    return np.asarray(
        residuals,
        dtype=np.float64,
    )


def build_jacobian_sparsity(
    node_count: int,
    edges: list[dict],
):
    row_count = 3 * len(edges)
    column_count = 3 * (node_count - 1)

    sparsity = lil_matrix(
        (row_count, column_count),
        dtype=np.int8,
    )

    for edge_index, edge in enumerate(edges):
        row_start = 3 * edge_index

        for node_id in (
            edge["source_id"],
            edge["target_id"],
        ):
            if node_id == 0:
                continue

            column_start = 3 * (node_id - 1)

            sparsity[
                row_start:row_start + 3,
                column_start:column_start + 3,
            ] = 1

    return sparsity.tocsr()


def optimize_pose_graph(
    nodes: list[dict],
    odometry_edges: list[dict],
    loop_constraint: dict | list[dict],
    odometry_translation_sigma_m: float,
    odometry_yaw_sigma_deg: float,
    loop_translation_sigma_m: float,
    loop_yaw_sigma_deg: float,
    maximum_evaluations: int,
    gradient_tolerance: float = 1e-5,
) -> tuple[list[dict], object]:
    if odometry_translation_sigma_m <= 0.0:
        raise ValueError(
            "Odometry translation sigma must be positive"
        )

    if loop_translation_sigma_m <= 0.0:
        raise ValueError(
            "Loop translation sigma must be positive"
        )

    if odometry_yaw_sigma_deg <= 0.0:
        raise ValueError(
            "Odometry yaw sigma must be positive"
        )

    if loop_yaw_sigma_deg <= 0.0:
        raise ValueError(
            "Loop yaw sigma must be positive"
        )

    if gradient_tolerance <= 0.0:
        raise ValueError(
            "Gradient tolerance must be positive"
        )

    loop_constraints = (
        loop_constraint
        if isinstance(loop_constraint, list)
        else [loop_constraint]
    )

    all_edges = [*odometry_edges, *loop_constraints]

    initial_state = nodes_to_state(nodes)

    odometry_yaw_sigma_rad = math.radians(
        odometry_yaw_sigma_deg
    )

    loop_yaw_sigma_rad = math.radians(
        loop_yaw_sigma_deg
    )

    initial_residual_vector = build_residual_vector(
        state=initial_state,
        fixed_node=nodes[0],
        edges=all_edges,
        odometry_translation_sigma_m=(
            odometry_translation_sigma_m
        ),
        odometry_yaw_sigma_rad=(
            odometry_yaw_sigma_rad
        ),
        loop_translation_sigma_m=(
            loop_translation_sigma_m
        ),
        loop_yaw_sigma_rad=(
            loop_yaw_sigma_rad
        ),
    )

    initial_cost = float(
        0.5
        * np.sum(
            initial_residual_vector
            * initial_residual_vector
        )
    )

    jacobian_sparsity = (
        build_jacobian_sparsity(
            node_count=len(nodes),
            edges=all_edges,
        )
    )

    result = least_squares(
        fun=build_residual_vector,
        x0=initial_state,
        jac_sparsity=jacobian_sparsity,
        args=(
            nodes[0],
            all_edges,
            odometry_translation_sigma_m,
            odometry_yaw_sigma_rad,
            loop_translation_sigma_m,
            loop_yaw_sigma_rad,
        ),
        method="trf",
        max_nfev=maximum_evaluations,
        gtol=gradient_tolerance,
        verbose=1,
    )

    optimized_poses = state_to_poses(
        result.x,
        nodes[0],
    )

    optimized_nodes: list[dict] = []

    for original_node, optimized_pose in zip(
        nodes,
        optimized_poses,
    ):
        optimized_nodes.append(
            {
                **original_node,
                "x": optimized_pose["x"],
                "y": optimized_pose["y"],
                "yaw_rad": wrap_angle(
                    optimized_pose["yaw_rad"]
                ),
            }
        )

    return optimized_nodes, result, initial_cost


def calculate_node_corrections(
    original_nodes: list[dict],
    optimized_nodes: list[dict],
) -> dict:
    position_corrections: list[float] = []
    yaw_corrections_deg: list[float] = []

    for original, optimized in zip(
        original_nodes,
        optimized_nodes,
    ):
        dx = optimized["x"] - original["x"]
        dy = optimized["y"] - original["y"]

        position_corrections.append(
            math.sqrt(dx * dx + dy * dy)
        )

        yaw_difference = wrap_angle(
            optimized["yaw_rad"]
            - original["yaw_rad"]
        )

        yaw_corrections_deg.append(
            abs(math.degrees(yaw_difference))
        )

    return {
        "mean_position_correction_m": float(
            np.mean(position_corrections)
        ),
        "maximum_position_correction_m": float(
            np.max(position_corrections)
        ),
        "mean_yaw_correction_deg": float(
            np.mean(yaw_corrections_deg)
        ),
        "maximum_yaw_correction_deg": float(
            np.max(yaw_corrections_deg)
        ),
    }


def loop_error_metrics(
    nodes: list[dict],
    loop_constraint: dict,
) -> dict:
    error = calculate_edge_error(
        nodes,
        loop_constraint,
    )

    translation_error = math.sqrt(
        error["dx"] ** 2
        + error["dy"] ** 2
    )

    return {
        "translation_error_m": translation_error,
        "yaw_error_deg": abs(
            math.degrees(error["dyaw_rad"])
        ),
    }


def write_optimized_nodes_csv(
    path: str | Path,
    original_nodes: list[dict],
    optimized_nodes: list[dict],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "keyframe_id",
        "timestamp",
        "x_before",
        "y_before",
        "yaw_before_rad",
        "yaw_before_deg",
        "x_after",
        "y_after",
        "yaw_after_rad",
        "yaw_after_deg",
        "position_correction_m",
        "yaw_correction_deg",
        "fixed",
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

        for original, optimized in zip(
            original_nodes,
            optimized_nodes,
        ):
            dx = optimized["x"] - original["x"]
            dy = optimized["y"] - original["y"]

            yaw_correction = wrap_angle(
                optimized["yaw_rad"]
                - original["yaw_rad"]
            )

            writer.writerow(
                {
                    "keyframe_id": original["keyframe_id"],
                    "timestamp": original["timestamp"],
                    "x_before": original["x"],
                    "y_before": original["y"],
                    "yaw_before_rad": original["yaw_rad"],
                    "yaw_before_deg": math.degrees(
                        original["yaw_rad"]
                    ),
                    "x_after": optimized["x"],
                    "y_after": optimized["y"],
                    "yaw_after_rad": optimized["yaw_rad"],
                    "yaw_after_deg": math.degrees(
                        optimized["yaw_rad"]
                    ),
                    "position_correction_m": math.sqrt(
                        dx * dx + dy * dy
                    ),
                    "yaw_correction_deg": math.degrees(
                        yaw_correction
                    ),
                    "fixed": int(original["fixed"]),
                }
            )


def write_optimization_summary(
    path: str | Path,
    original_nodes: list[dict],
    optimized_nodes: list[dict],
    odometry_edges: list[dict],
    loop_constraint: dict,
    solver_result,
    settings: dict,
    initial_cost: float,
) -> None:

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corrections = calculate_node_corrections(
        original_nodes,
        optimized_nodes,
    )

    loop_before = loop_error_metrics(
        original_nodes,
        loop_constraint,
    )

    loop_after = loop_error_metrics(
        optimized_nodes,
        loop_constraint,
    )

    summary = {
        "node_count": len(original_nodes),
        "odometry_edge_count": len(odometry_edges),
        "loop_edge_count": 1,
        "loop_source_id": loop_constraint["source_id"],
        "loop_target_id": loop_constraint["target_id"],
        "solver_success": bool(solver_result.success),
        "solver_status": int(solver_result.status),
        "solver_message": str(solver_result.message),
        "solver_function_evaluations": int(
            solver_result.nfev
        ),
        "solver_optimality": float(
            solver_result.optimality
        ),
        "initial_cost": float(initial_cost),
        "final_cost": float(solver_result.cost),
        "loop_translation_error_before_m": (
            loop_before["translation_error_m"]
        ),
        "loop_translation_error_after_m": (
            loop_after["translation_error_m"]
        ),
        "loop_yaw_error_before_deg": (
            loop_before["yaw_error_deg"]
        ),
        "loop_yaw_error_after_deg": (
            loop_after["yaw_error_deg"]
        ),
        **corrections,
        "settings": settings,
    }

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def plot_optimized_graph(
    path: str | Path,
    original_nodes: list[dict],
    optimized_nodes: list[dict],
    loop_constraint: dict,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_x = [
        node["x"]
        for node in original_nodes
    ]

    before_y = [
        node["y"]
        for node in original_nodes
    ]

    after_x = [
        node["x"]
        for node in optimized_nodes
    ]

    after_y = [
        node["y"]
        for node in optimized_nodes
    ]

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    axis.plot(
        before_x,
        before_y,
        linewidth=1.0,
        label="Before loop optimization",
    )

    axis.plot(
        after_x,
        after_y,
        linewidth=1.5,
        label="After loop optimization",
    )

    source_id = loop_constraint["source_id"]
    target_id = loop_constraint["target_id"]

    source_node = optimized_nodes[source_id]
    target_node = optimized_nodes[target_id]

    axis.plot(
        [
            source_node["x"],
            target_node["x"],
        ],
        [
            source_node["y"],
            target_node["y"],
        ],
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Loop edge "
            f"{source_id} ↔ {target_id}"
        ),
    )

    axis.scatter(
        [optimized_nodes[0]["x"]],
        [optimized_nodes[0]["y"]],
        s=80,
        marker="o",
        label="Fixed first node",
    )

    axis.scatter(
        [optimized_nodes[-1]["x"]],
        [optimized_nodes[-1]["y"]],
        s=80,
        marker="x",
        label="Final node",
    )

    axis.set_title(
        "Pose graph before and after loop optimization"
    )

    axis.set_xlabel("Local X [m]")
    axis.set_ylabel("Local Y [m]")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)
