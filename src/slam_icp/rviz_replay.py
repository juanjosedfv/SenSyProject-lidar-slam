"""Replay a completed offline SLAM run as ROS 2 topics for RViz."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from slam_icp.geometry import pose_to_transform
from slam_icp.io import read_trajectory_csv_compatible
from slam_icp.keyframes import select_keyframes
from slam_icp.outputs import (
    calculate_common_level_transform_from_poses,
    load_ascii_pcd,
)
from slam_icp.trajectory import TimedPose, rotation_matrix_to_quaternion


@dataclass(frozen=True)
class ReplayArtifacts:
    """Paths and metadata belonging to one completed SLAM run."""

    run_directory: Path
    internal_directory: Path
    repository_root: Path
    summary: dict[str, Any]
    configuration: dict[str, Any]
    corrected_trajectory: Path
    original_trajectory: Path
    map_path: Path
    fixed_frame: str


def _repository_root() -> Path:
    """Return the repository root from src/slam_icp/rviz_replay.py."""

    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return value


def _resolve_run_directory(path: str | Path, repository_root: Path) -> Path:
    """Resolve a user-provided run directory."""

    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        current_candidate = (Path.cwd() / candidate).resolve()
        repository_candidate = (repository_root / candidate).resolve()

        resolved = (
            current_candidate
            if current_candidate.exists()
            else repository_candidate
        )

    if not resolved.exists():
        raise FileNotFoundError(f"SLAM run directory was not found: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"SLAM run path is not a directory: {resolved}")

    return resolved


def _resolve_configuration_path(
    value: str | Path,
    repository_root: Path,
) -> Path:
    """Resolve a path stored inside effective_config.yaml."""

    candidate = Path(value).expanduser()

    if not candidate.is_absolute():
        candidate = repository_root / candidate

    return candidate.resolve()


def _first_existing_path(
    configured_value: str | None,
    repository_root: Path,
    fallback: Path,
    label: str,
) -> Path:
    """Use a stored path when valid, otherwise use a run-folder fallback."""

    candidates: list[Path] = []

    if configured_value:
        configured_path = Path(configured_value).expanduser()

        if not configured_path.is_absolute():
            configured_path = repository_root / configured_path

        candidates.append(configured_path.resolve())

    candidates.append(fallback.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"{label} was not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def discover_run_artifacts(run_directory: str | Path) -> ReplayArtifacts:
    """Discover the files belonging to a completed SLAM run."""

    repository_root = _repository_root()
    run_path = _resolve_run_directory(run_directory, repository_root)
    internal_path = run_path / "_internal"

    summary_path = internal_path / "run_summary.json"
    configuration_path = internal_path / "effective_config.yaml"

    summary = _read_json(summary_path)

    if not configuration_path.exists():
        raise FileNotFoundError(
            f"Effective SLAM configuration was not found: {configuration_path}"
        )

    with configuration_path.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)

    if not isinstance(configuration, dict):
        raise ValueError(
            f"Expected a YAML mapping in: {configuration_path}"
        )

    corrected_trajectory = _first_existing_path(
        configured_value=summary.get("corrected_trajectory_path"),
        repository_root=repository_root,
        fallback=internal_path / "trajectory_slam_local.csv",
        label="Corrected SLAM trajectory",
    )

    map_path = _first_existing_path(
        configured_value=summary.get("map_path"),
        repository_root=repository_root,
        fallback=run_path / "map_slam.pcd",
        label="SLAM map",
    )

    try:
        original_value = configuration["inputs"]["trajectory_csv"]
        fixed_frame = configuration["frames"]["parent_frame"]
    except KeyError as error:
        raise ValueError(
            "effective_config.yaml is missing inputs.trajectory_csv "
            "or frames.parent_frame"
        ) from error

    original_trajectory = _resolve_configuration_path(
        original_value,
        repository_root,
    )

    if not original_trajectory.exists():
        raise FileNotFoundError(
            f"Original ICP trajectory was not found: {original_trajectory}"
        )

    return ReplayArtifacts(
        run_directory=run_path,
        internal_directory=internal_path,
        repository_root=repository_root,
        summary=summary,
        configuration=configuration,
        corrected_trajectory=corrected_trajectory,
        original_trajectory=original_trajectory,
        map_path=map_path,
        fixed_frame=str(fixed_frame),
    )


def _poses_to_dictionaries(poses: list[TimedPose]) -> list[dict[str, Any]]:
    """Convert canonical poses to the dictionaries used by keyframe selection."""

    return [
        {
            "pose_index": index,
            "timestamp": pose.timestamp,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "qx": pose.qx,
            "qy": pose.qy,
            "qz": pose.qz,
            "qw": pose.qw,
            "backend": pose.backend,
            "parent_frame": pose.parent_frame,
            "child_frame": pose.child_frame,
        }
        for index, pose in enumerate(poses)
    ]


def _transform_pose(pose: TimedPose, left_transform: np.ndarray) -> TimedPose:
    """Apply a common world-frame transform to one pose."""

    transformed = left_transform @ pose_to_transform(pose)

    qx, qy, qz, qw = rotation_matrix_to_quaternion(
        transformed[:3, :3]
    )

    return TimedPose(
        timestamp=pose.timestamp,
        x=float(transformed[0, 3]),
        y=float(transformed[1, 3]),
        z=float(transformed[2, 3]),
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        backend=pose.backend,
        parent_frame=pose.parent_frame,
        child_frame=pose.child_frame,
    )


def _time_message(timestamp: float) -> TimeMessage:
    """Convert floating-point seconds to a ROS Time message."""

    seconds = int(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))

    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000

    return TimeMessage(sec=seconds, nanosec=nanoseconds)


def _pose_stamped(
    pose: TimedPose,
    frame_id: str,
    stamp: TimeMessage | None = None,
) -> PoseStamped:
    """Convert a canonical pose to geometry_msgs/PoseStamped."""

    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp = stamp or _time_message(pose.timestamp)

    message.pose.position.x = pose.x
    message.pose.position.y = pose.y
    message.pose.position.z = pose.z

    message.pose.orientation.x = pose.qx
    message.pose.orientation.y = pose.qy
    message.pose.orientation.z = pose.qz
    message.pose.orientation.w = pose.qw

    return message


def _path_message(
    poses: list[TimedPose],
    frame_id: str,
    header_stamp: TimeMessage,
) -> PathMessage:
    """Create a complete ROS Path message."""

    message = PathMessage()
    message.header.frame_id = frame_id
    message.header.stamp = header_stamp

    message.poses = [
        _pose_stamped(pose, frame_id)
        for pose in poses
    ]

    return message


def _xyz_pointcloud_message(
    points: np.ndarray,
    frame_id: str,
    stamp: TimeMessage,
) -> PointCloud2:
    """Convert Nx3 XYZ values to sensor_msgs/PointCloud2."""

    values = np.asarray(points, dtype=np.float32)
    values = values[np.isfinite(values).all(axis=1)]
    values = np.ascontiguousarray(values)

    message = PointCloud2()
    message.header = Header(stamp=stamp, frame_id=frame_id)

    message.height = 1
    message.width = len(values)

    message.fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]

    message.is_bigendian = False
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.data = values.tobytes()
    message.is_dense = True

    return message


def _point(x: float, y: float, z: float) -> Point:
    message = Point()
    message.x = float(x)
    message.y = float(y)
    message.z = float(z)
    return message


class SlamReplayNode(Node):
    """Publish one completed offline SLAM result for RViz."""

    def __init__(
        self,
        run_directory: str | Path,
        replay_rate: float = 1.0,
        repeat: bool = False,
    ) -> None:
        super().__init__("slam_rviz_replay")

        if replay_rate <= 0.0:
            raise ValueError("Replay rate must be greater than zero")

        self.replay_rate = float(replay_rate)
        self.repeat = bool(repeat)

        self.artifacts = discover_run_artifacts(run_directory)
        self.fixed_frame = self.artifacts.fixed_frame
        self.vehicle_frame = "slam_vehicle"

        original_raw = read_trajectory_csv_compatible(
            self.artifacts.original_trajectory
        )

        corrected_raw = read_trajectory_csv_compatible(
            self.artifacts.corrected_trajectory
        )

        if not original_raw:
            raise ValueError("The original ICP trajectory is empty")

        if not corrected_raw:
            raise ValueError("The corrected SLAM trajectory is empty")

        if len(original_raw) != len(corrected_raw):
            raise ValueError(
                "Original and corrected trajectories have different pose counts: "
                f"{len(original_raw)} and {len(corrected_raw)}"
            )

        original_times = np.asarray(
            [pose.timestamp for pose in original_raw],
            dtype=np.float64,
        )

        corrected_times = np.asarray(
            [pose.timestamp for pose in corrected_raw],
            dtype=np.float64,
        )

        if not np.allclose(
            original_times,
            corrected_times,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Original and corrected trajectory timestamps do not match"
            )

        mapping_configuration = self.artifacts.configuration.get(
            "mapping",
            {},
        )

        leveling_enabled = bool(
            mapping_configuration.get("leveling_enabled", False)
        )

        if leveling_enabled:
            leveling_transform, leveling_information = (
                calculate_common_level_transform_from_poses(
                    corrected_pose_times=corrected_times,
                    reference_poses=_poses_to_dictionaries(original_raw),
                )
            )

            self.get_logger().info(
                "Applying the map's existing leveling transform to RViz paths"
            )

            if leveling_information:
                tilt = leveling_information.get("tilt_deg")

                if tilt is not None:
                    self.get_logger().info(
                        f"Leveling tilt: {float(tilt):.2f} deg"
                    )
        else:
            leveling_transform = np.eye(4, dtype=np.float64)

        self.original_poses = [
            _transform_pose(pose, leveling_transform)
            for pose in original_raw
        ]

        self.corrected_poses = [
            _transform_pose(pose, leveling_transform)
            for pose in corrected_raw
        ]

        keyframe_configuration = self.artifacts.configuration.get(
            "keyframes",
            {},
        )

        selected_keyframes = select_keyframes(
            _poses_to_dictionaries(original_raw),
            translation_threshold_m=float(
                keyframe_configuration.get(
                    "translation_threshold_m",
                    1.0,
                )
            ),
            yaw_threshold_deg=float(
                keyframe_configuration.get(
                    "yaw_threshold_deg",
                    10.0,
                )
            ),
            time_threshold_s=float(
                keyframe_configuration.get(
                    "time_threshold_s",
                    2.0,
                )
            ),
        )

        self.keyframes = selected_keyframes

        self.keyframe_pose_indices = {
            int(keyframe["keyframe_id"]): int(keyframe["pose_index"])
            for keyframe in selected_keyframes
        }

        self.accepted_loops = self.artifacts.summary.get(
            "accepted_loop_ids",
            [],
        )

        map_points = load_ascii_pcd(self.artifacts.map_path)

        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        dynamic_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.map_publisher = self.create_publisher(
            PointCloud2,
            "/slam/map",
            static_qos,
        )

        self.original_path_publisher = self.create_publisher(
            PathMessage,
            "/slam/original_path",
            static_qos,
        )

        self.corrected_path_publisher = self.create_publisher(
            PathMessage,
            "/slam/corrected_path",
            static_qos,
        )

        self.current_pose_publisher = self.create_publisher(
            PoseStamped,
            "/slam/current_pose",
            dynamic_qos,
        )

        self.keyframe_publisher = self.create_publisher(
            MarkerArray,
            "/slam/keyframes",
            static_qos,
        )

        self.loop_publisher = self.create_publisher(
            MarkerArray,
            "/slam/loop_edges",
            static_qos,
        )

        self.transform_broadcaster = TransformBroadcaster(self)

        initial_stamp = self.get_clock().now().to_msg()

        self.map_message = _xyz_pointcloud_message(
            map_points,
            self.fixed_frame,
            initial_stamp,
        )

        self.original_path_message = _path_message(
            self.original_poses,
            self.fixed_frame,
            initial_stamp,
        )

        self.keyframe_markers = self._build_keyframe_markers(
            initial_stamp
        )

        self.loop_markers = self._build_loop_markers(
            initial_stamp
        )

        self.corrected_path_message = PathMessage()
        self.corrected_path_message.header.frame_id = self.fixed_frame
        self.corrected_path_message.header.stamp = initial_stamp

        self.current_index = 0
        self.replay_start_wall_time = time.monotonic()
        self.finished_wall_time: float | None = None

        self._publish_static_messages()

        self.timer = self.create_timer(
            0.02,
            self._timer_callback,
        )

        self.get_logger().info(
            f"Run directory: {self.artifacts.run_directory}"
        )
        self.get_logger().info(
            f"Map points: {len(map_points)}"
        )
        self.get_logger().info(
            f"Trajectory poses: {len(self.corrected_poses)}"
        )
        self.get_logger().info(
            f"Keyframes: {len(self.keyframes)}"
        )
        self.get_logger().info(
            f"Accepted loops: {len(self.accepted_loops)}"
        )
        self.get_logger().info(
            f"Replay rate: {self.replay_rate:.2f}x"
        )

    def _publish_static_messages(self) -> None:
        self.map_publisher.publish(self.map_message)
        self.original_path_publisher.publish(
            self.original_path_message
        )
        self.keyframe_publisher.publish(
            self.keyframe_markers
        )
        self.loop_publisher.publish(
            self.loop_markers
        )

    def _build_keyframe_markers(
        self,
        stamp: TimeMessage,
    ) -> MarkerArray:
        marker = Marker()

        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = stamp

        marker.ns = "slam_keyframes"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.55
        marker.scale.y = 0.55
        marker.scale.z = 0.55

        marker.color.r = 0.10
        marker.color.g = 0.80
        marker.color.b = 1.00
        marker.color.a = 1.00

        for keyframe in self.keyframes:
            pose_index = int(keyframe["pose_index"])
            pose = self.corrected_poses[pose_index]

            marker.points.append(
                _point(pose.x, pose.y, pose.z)
            )

        return MarkerArray(markers=[marker])

    def _build_loop_markers(
        self,
        stamp: TimeMessage,
    ) -> MarkerArray:
        marker = Marker()

        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = stamp

        marker.ns = "slam_loop_edges"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25

        marker.color.r = 1.00
        marker.color.g = 0.10
        marker.color.b = 0.10
        marker.color.a = 1.00

        for loop in self.accepted_loops:
            source_id = int(loop["source_id"])
            target_id = int(loop["target_id"])

            if source_id not in self.keyframe_pose_indices:
                self.get_logger().warning(
                    f"Loop source keyframe was not found: {source_id}"
                )
                continue

            if target_id not in self.keyframe_pose_indices:
                self.get_logger().warning(
                    f"Loop target keyframe was not found: {target_id}"
                )
                continue

            source_pose = self.corrected_poses[
                self.keyframe_pose_indices[source_id]
            ]

            target_pose = self.corrected_poses[
                self.keyframe_pose_indices[target_id]
            ]

            marker.points.append(
                _point(
                    source_pose.x,
                    source_pose.y,
                    source_pose.z,
                )
            )

            marker.points.append(
                _point(
                    target_pose.x,
                    target_pose.y,
                    target_pose.z,
                )
            )

        return MarkerArray(markers=[marker])

    def _publish_current_pose(
        self,
        pose: TimedPose,
        stamp: TimeMessage,
    ) -> None:
        pose_message = _pose_stamped(
            pose,
            self.fixed_frame,
            stamp,
        )

        self.current_pose_publisher.publish(
            pose_message
        )

        transform = TransformStamped()

        transform.header.frame_id = self.fixed_frame
        transform.header.stamp = stamp
        transform.child_frame_id = self.vehicle_frame

        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.translation.z = pose.z

        transform.transform.rotation.x = pose.qx
        transform.transform.rotation.y = pose.qy
        transform.transform.rotation.z = pose.qz
        transform.transform.rotation.w = pose.qw

        self.transform_broadcaster.sendTransform(
            transform
        )

    def _reset_replay(self) -> None:
        self.current_index = 0
        self.replay_start_wall_time = time.monotonic()
        self.finished_wall_time = None

        self.corrected_path_message = PathMessage()
        self.corrected_path_message.header.frame_id = self.fixed_frame
        self.corrected_path_message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.corrected_path_publisher.publish(
            self.corrected_path_message
        )

        self.get_logger().info("Replay restarted")

    def _timer_callback(self) -> None:
        if self.finished_wall_time is not None:
            if (
                self.repeat
                and time.monotonic() - self.finished_wall_time >= 1.0
            ):
                self._reset_replay()

            return

        first_timestamp = self.corrected_poses[0].timestamp

        elapsed_replay_time = (
            time.monotonic() - self.replay_start_wall_time
        ) * self.replay_rate

        target_timestamp = (
            first_timestamp + elapsed_replay_time
        )

        newest_pose: TimedPose | None = None

        while (
            self.current_index < len(self.corrected_poses)
            and self.corrected_poses[
                self.current_index
            ].timestamp <= target_timestamp
        ):
            newest_pose = self.corrected_poses[
                self.current_index
            ]

            self.corrected_path_message.poses.append(
                _pose_stamped(
                    newest_pose,
                    self.fixed_frame,
                )
            )

            self.current_index += 1

        if newest_pose is not None:
            current_stamp = self.get_clock().now().to_msg()

            self.corrected_path_message.header.stamp = (
                current_stamp
            )

            self.corrected_path_publisher.publish(
                self.corrected_path_message
            )

            self._publish_current_pose(
                newest_pose,
                current_stamp,
            )

        if self.current_index >= len(self.corrected_poses):
            self.finished_wall_time = time.monotonic()

            self.get_logger().info(
                "SLAM trajectory replay completed"
            )
