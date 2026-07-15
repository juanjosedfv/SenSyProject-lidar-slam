"""Adapter: read KISS-ICP `/kiss/odometry` from a recorded ROS 2 bag.

Converts the recorded odometry into the same ``OdometryStep`` list that the
existing pipeline already consumes, so all downstream code (trajectory_xyz,
align_2d_rigid, evaluation, CSV/plot writers) keeps working unchanged.

IMPORTANT: poses are timestamped with ``header.stamp`` (the original sensor
time inherited from /ouster/points), NOT the bag receive time (which is the
wall clock of when you recorded, i.e. the wrong epoch entirely).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .odometry import OdometryStep, identity_transform


def find_bag_db(bag_dir: str | Path) -> Path:
    """Locate the single .db3 file inside a ROS 2 bag directory."""

    bag_path = Path(bag_dir)
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path
    candidates = sorted(bag_path.glob("*.db3"))
    if not candidates:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found {len(candidates)} .db3 files in {bag_path}, expected one: "
            + ", ".join(p.name for p in candidates)
        )
    return candidates[0]


def quaternion_to_rotation(x: float, y: float, z: float, w: float):
    """Convert a WXYZ-normalised quaternion into a 3x3 rotation matrix."""

    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm < 1e-12:
        return np.eye(3, dtype="float64")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype="float64",
    )


def _read_raw_messages(db_path: Path, topic_name: str):
    """Return (bag_receive_time_ns, serialized_bytes) rows for one topic."""

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            select messages.timestamp, messages.data
            from messages
            join topics on messages.topic_id = topics.id
            where topics.name = ?
            order by messages.timestamp
            """,
            (topic_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"No messages for topic '{topic_name}' in {db_path}")
    return rows


def load_kiss_odometry(
    bag_dir: str | Path,
    topic_name: str = "/kiss/odometry",
    verbose: bool = True,
) -> list[OdometryStep]:
    """Load recorded KISS-ICP odometry as a list of ``OdometryStep``.

    Requires a sourced ROS 2 environment (rclpy + nav_msgs) for deserialization.
    """

    try:
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading /kiss/odometry needs a sourced ROS 2 environment. Run:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source ~/ros2_ws/install/setup.bash"
        ) from exc

    db_path = find_bag_db(bag_dir)
    rows = _read_raw_messages(db_path, topic_name)

    trajectory: list[OdometryStep] = []
    for receive_time_ns, blob in rows:
        msg = deserialize_message(bytes(blob), Odometry)

        # Sensor time inherited from /ouster/points -- the only stamp that can
        # be matched against the GNSS ground truth.
        stamp = msg.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        transform = identity_transform()
        transform[:3, :3] = quaternion_to_rotation(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        transform[0, 3] = float(position.x)
        transform[1, 3] = float(position.y)
        transform[2, 3] = float(position.z)

        trajectory.append(
            OdometryStep(
                timestamp_s=timestamp_s,
                source_file=f"{topic_name}@{int(receive_time_ns)}",
                transform=transform,
                fitness=1.0,  # KISS-ICP does not publish per-scan fitness
                inlier_rmse=float("nan"),  # downstream filters with np.isfinite
            )
        )

    if verbose:
        _report(trajectory, rows, db_path, topic_name)
    return trajectory


def _report(trajectory, rows, db_path: Path, topic_name: str) -> None:
    """Print a short sanity report, including the header-vs-bag-time gap."""

    header_t = np.array([step.timestamp_s for step in trajectory], dtype="float64")
    receive_t = np.array([row[0] for row in rows], dtype="float64") * 1e-9

    print(f"KISS-ICP odometry: {len(trajectory)} poses from {db_path.name} [{topic_name}]")
    print(f"  header.stamp range : {header_t[0]:.3f} .. {header_t[-1]:.3f} s")
    print(f"  data duration      : {header_t[-1] - header_t[0]:.1f} s")
    print(f"  bag receive range  : {receive_t[0]:.3f} .. {receive_t[-1]:.3f} s")

    offset = receive_t[0] - header_t[0]
    if abs(offset) > 60.0:
        print(
            f"  note: bag receive time is {offset / 86400.0:.1f} days after the sensor "
            "stamp (recorded offline). Using header.stamp, as required for GNSS matching."
        )

    steps = np.diff(header_t)
    if len(steps) and (steps <= 0).any():
        print(f"  WARNING: {int((steps <= 0).sum())} non-increasing timestamps")
    if len(steps):
        print(f"  mean scan rate     : {1.0 / float(np.mean(steps)):.2f} Hz")


def trajectory_time_bounds(trajectory: list[OdometryStep]) -> tuple[float, float]:
    """Return (first, last) sensor timestamp in seconds."""

    return trajectory[0].timestamp_s, trajectory[-1].timestamp_s