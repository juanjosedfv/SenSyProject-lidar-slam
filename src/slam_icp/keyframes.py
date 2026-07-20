from __future__ import annotations

import csv                  #Reads and writes CSV files.
import math                 # Provides: square root, atan2, sine and cosine, conversion between radians and degrees.
import json
from pathlib import Path

import matplotlib.pyplot as plt

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


## STEP 1: Load the trajectory from a CSV file and validate its contents.
""" 
opens trajectory_local.csv;
checks that the required columns exist;
converts text values into Python numbers;
checks that timestamps always increase;
returns all poses as a list --> list[dict]
"""
def load_trajectory(path: str | Path) -> list[dict]:
    trajectory_path = Path(path)

    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Trajectory file does not exist: {trajectory_path}"
        )

    poses: list[dict] = []

    with trajectory_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Trajectory file has no CSV header: {trajectory_path}"
            )

        missing_columns = (
            REQUIRED_TRAJECTORY_COLUMNS - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Trajectory file is missing required columns: "
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
                    f"Invalid trajectory values in CSV row "
                    f"{pose_index + 2}"
                ) from error

            poses.append(pose)

    if not poses:
        raise ValueError(
            f"No trajectory poses were found in {trajectory_path}"
        )

    non_increasing_timestamps = 0

    for previous_pose, current_pose in zip(
        poses[:-1],
        poses[1:],
    ):
        if current_pose["timestamp"] <= previous_pose["timestamp"]:
            non_increasing_timestamps += 1

    if non_increasing_timestamps > 0:
        raise ValueError(
            "Trajectory contains "
            f"{non_increasing_timestamps} non-increasing timestamps"
        )

    return poses

## STEP 2: Convert quaternions into yaw angles.
"""
The LiDAR orientation is stored as a quaternion: qx, qy, qz, qw
This function converts the quaternion into the yaw angle.
Yaw means rotation around the vertical axis: turning left or right
The quaternion is normalized first to prevent errors if its length is slightly different from one.
"""
def quaternion_to_yaw(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> float:
    quaternion_norm = math.sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    )

    if quaternion_norm == 0.0:
        raise ValueError("Quaternion has zero length")

    qx /= quaternion_norm
    qy /= quaternion_norm
    qz /= quaternion_norm
    qw /= quaternion_norm

    sin_yaw = 2.0 * (
        qw * qz
        + qx * qy
    )

    cos_yaw = 1.0 - 2.0 * (
        qy * qy
        + qz * qz
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )

## STEP 3: Calculate the difference between two angles
"""
This calculates the difference between two angles while handling the transition between: +180° and -180°
Without angle wrapping, a change from 179° to -179° would incorrectly appear to be 358°. 
The correct difference is 2°.
"""
def wrapped_angle_difference(
    current_angle: float,
    reference_angle: float,
) -> float:
    raw_difference = current_angle - reference_angle

    return math.atan2(
        math.sin(raw_difference),
        math.cos(raw_difference),
    )

## STEP 4: Calculate the horizontal distance between two poses
"""
This calculates the distance between two poses using only: x and y
We ignore z because the XTrack is a ground vehicle 
and small vertical changes should not generate unnecessary keyframes.
"""
def horizontal_distance(
    first_pose: dict,
    second_pose: dict,
) -> float:
    dx = second_pose["x"] - first_pose["x"]
    dy = second_pose["y"] - first_pose["y"]

    return math.sqrt(
        dx * dx
        + dy * dy
    )

## STEP 5: Calculate the yaw change between two poses
"""
This function:
1. converts both quaternions into yaw angles;
2. calculates the wrapped difference;
3. converts the result from radians into degrees.
"""
def yaw_change_degrees(
    first_pose: dict,
    second_pose: dict,
) -> float:
    first_yaw = quaternion_to_yaw(
        first_pose["qx"],
        first_pose["qy"],
        first_pose["qz"],
        first_pose["qw"],
    )

    second_yaw = quaternion_to_yaw(
        second_pose["qx"],
        second_pose["qy"],
        second_pose["qz"],
        second_pose["qw"],
    )

    yaw_difference = wrapped_angle_difference(
        second_yaw,
        first_yaw,
    )

    return abs(
        math.degrees(yaw_difference)
    )

## STEP 6: Calculate the time difference between two poses
""" This calculates how much time passed between two poses
"""
def time_difference_seconds(
    first_pose: dict,
    second_pose: dict,
) -> float:
    return (
        second_pose["timestamp"]
        - first_pose["timestamp"]
    )

## STEP 7: Create a keyframe dictionary from a pose
"""
This copies an original trajectory pose and adds keyframe information:

1. keyframe ID
2. yaw angle
3. selection reason
4. distance from previous keyframe
5. yaw change
6. time change

It prevents the keyframe-selection loop from repeating the same dictionary-building code.
"""
def create_keyframe(
    pose: dict,
    keyframe_id: int,
    selection_reason: str,
    distance_from_last_m: float,
    yaw_change_from_last_deg: float,
    time_from_last_s: float,
) -> dict:
    keyframe = pose.copy()

    keyframe["keyframe_id"] = keyframe_id
    keyframe["selection_reason"] = selection_reason

    keyframe["distance_from_last_m"] = (
        distance_from_last_m
    )

    keyframe["yaw_change_from_last_deg"] = (
        yaw_change_from_last_deg
    )

    keyframe["time_from_last_s"] = (
        time_from_last_s
    )

    keyframe["yaw_deg"] = math.degrees(
        quaternion_to_yaw(
            pose["qx"],
            pose["qy"],
            pose["qz"],
            pose["qw"],
        )
    )

    return keyframe

## STEP 8: Select keyframes from a trajectory based on translation, rotation, and time thresholds
"""
main algorithm, it:
1. always selects the first pose;
2. compares each candidate against the last selected keyframe;
3. selects a new keyframe when translation, rotation, or time exceeds its threshold;
4. always includes the final pose.
"""

def select_keyframes(
    poses: list[dict],
    translation_threshold_m: float = 1.0,
    yaw_threshold_deg: float = 10.0,
    time_threshold_s: float = 2.0,
) -> list[dict]:
    if not poses:
        raise ValueError(
            "Cannot select keyframes from an empty trajectory"
        )

    if translation_threshold_m <= 0.0:
        raise ValueError(
            "Translation threshold must be greater than zero"
        )

    if yaw_threshold_deg <= 0.0:
        raise ValueError(
            "Yaw threshold must be greater than zero"
        )

    if time_threshold_s <= 0.0:
        raise ValueError(
            "Time threshold must be greater than zero"
        )

    first_keyframe = create_keyframe(
        pose=poses[0],
        keyframe_id=0,
        selection_reason="first",
        distance_from_last_m=0.0,
        yaw_change_from_last_deg=0.0,
        time_from_last_s=0.0,
    )

    keyframes = [first_keyframe]

    last_selected_pose = poses[0]

    for current_pose in poses[1:-1]:
        distance_m = horizontal_distance(
            last_selected_pose,
            current_pose,
        )

        yaw_change_deg = yaw_change_degrees(
            last_selected_pose,
            current_pose,
        )

        time_change_s = time_difference_seconds(
            last_selected_pose,
            current_pose,
        )

        selection_reasons: list[str] = []

        if distance_m >= translation_threshold_m:
            selection_reasons.append("translation")

        if yaw_change_deg >= yaw_threshold_deg:
            selection_reasons.append("rotation")

        if time_change_s >= time_threshold_s:
            selection_reasons.append("time")

        if not selection_reasons:
            continue

        keyframe = create_keyframe(
            pose=current_pose,
            keyframe_id=len(keyframes),
            selection_reason="+".join(selection_reasons),
            distance_from_last_m=distance_m,
            yaw_change_from_last_deg=yaw_change_deg,
            time_from_last_s=time_change_s,
        )

        keyframes.append(keyframe)
        last_selected_pose = current_pose

    final_pose = poses[-1]

    if final_pose["pose_index"] != keyframes[-1]["pose_index"]:
        distance_m = horizontal_distance(
            last_selected_pose,
            final_pose,
        )

        yaw_change_deg = yaw_change_degrees(
            last_selected_pose,
            final_pose,
        )

        time_change_s = time_difference_seconds(
            last_selected_pose,
            final_pose,
        )

        final_keyframe = create_keyframe(
            pose=final_pose,
            keyframe_id=len(keyframes),
            selection_reason="final",
            distance_from_last_m=distance_m,
            yaw_change_from_last_deg=yaw_change_deg,
            time_from_last_s=time_change_s,
        )

        keyframes.append(final_keyframe)

    return keyframes

# STEP 9: Write the selected keyframes to a CSV file: keyframes.csv

def write_keyframes_csv(
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
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "yaw_deg",
        "backend",
        "parent_frame",
        "child_frame",
        "distance_from_last_m",
        "yaw_change_from_last_deg",
        "time_from_last_s",
        "selection_reason",
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
                    field: keyframe.get(field, "")
                    for field in fieldnames
                }
            )

def calculate_mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)

## STEP 10: Write a summary of the keyframe selection process to a JSON file: keyframes_summary.json
"""
This creates a JSON report containing:
1. number of input poses
2. number of keyframes
3. average distance between keyframes
4. average yaw change
5. average time difference
6. first and last timestamp
"""
def write_keyframe_summary(
    path: str | Path,
    input_pose_count: int,
    keyframes: list[dict],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distances = [
        float(keyframe["distance_from_last_m"])
        for keyframe in keyframes[1:]
    ]

    yaw_changes = [
        float(keyframe["yaw_change_from_last_deg"])
        for keyframe in keyframes[1:]
    ]

    time_changes = [
        float(keyframe["time_from_last_s"])
        for keyframe in keyframes[1:]
    ]

    summary = {
        "backend": keyframes[0]["backend"],
        "input_pose_count": input_pose_count,
        "keyframe_count": len(keyframes),
        "keyframe_percentage": (
            100.0
            * len(keyframes)
            / input_pose_count
        ),
        "mean_distance_between_keyframes_m": (
            calculate_mean(distances)
        ),
        "maximum_distance_between_keyframes_m": (
            max(distances)
            if distances
            else 0.0
        ),
        "mean_yaw_change_between_keyframes_deg": (
            calculate_mean(yaw_changes)
        ),
        "maximum_yaw_change_between_keyframes_deg": (
            max(yaw_changes)
            if yaw_changes
            else 0.0
        ),
        "mean_time_between_keyframes_s": (
            calculate_mean(time_changes)
        ),
        "maximum_time_between_keyframes_s": (
            max(time_changes)
            if time_changes
            else 0.0
        ),
        "first_timestamp": keyframes[0]["timestamp"],
        "last_timestamp": keyframes[-1]["timestamp"],
    }

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

## STEP 11: Plot the trajectory and selected keyframes
"""
This function creates a 2D plot of the full trajectory and the selected keyframes.
It saves the plot as a PNG file:
1. the complete KISS trajectory;
2. all selected keyframes;
3. the start;
4. the end;
5. some keyframe IDs.
This plot will tell us whether the selection thresholds are sensible.
"""
def plot_keyframes(
    path: str | Path,
    poses: list[dict],
    keyframes: list[dict],
    label_every: int = 20,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_x = [
        pose["x"]
        for pose in poses
    ]

    trajectory_y = [
        pose["y"]
        for pose in poses
    ]

    keyframe_x = [
        keyframe["x"]
        for keyframe in keyframes
    ]

    keyframe_y = [
        keyframe["y"]
        for keyframe in keyframes
    ]

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.plot(
        trajectory_x,
        trajectory_y,
        linewidth=1.0,
        label="Full odometry trajectory",
    )

    axis.scatter(
        keyframe_x,
        keyframe_y,
        s=12,
        label="Selected keyframes",
    )

    axis.scatter(
        [keyframes[0]["x"]],
        [keyframes[0]["y"]],
        s=70,
        marker="o",
        label="Start",
    )

    axis.scatter(
        [keyframes[-1]["x"]],
        [keyframes[-1]["y"]],
        s=70,
        marker="x",
        label="End",
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
        "Selected SLAM keyframes"
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