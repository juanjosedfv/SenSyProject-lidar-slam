"""Build the global 3D point-cloud map from the recorded KISS-ICP poses.

Streams /ouster/points out of the original rosbag, looks up the matching
KISS-ICP pose by header stamp, transforms each scan into the odometry frame and
accumulates everything into a voxel grid. No re-run of KISS-ICP needed, and the
map is guaranteed to be consistent with the trajectory CSV, because both come
from the same recorded poses.

Reuses the team's ``preprocess.filter_points`` (range filtering) and
``odometry.write_ascii_pcd`` (PCD writer).

Run from the repository root with ROS 2 sourced:

    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 -m src.build_map \
        --bag /home/jinghao/ros2_ws/data/rosbag \
        --kiss-bag /home/jinghao/ros2_ws/data/kiss_output
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = ""

from baseline_icp.odometry import write_ascii_pcd
from kiss_icp.adapter import find_bag_db, load_kiss_odometry
from level import describe, level_transform
from preprocess import filter_points
from slam_icp.trajectory import from_odometry_steps, timed_pose_to_transform, write_trajectory_csv

# sensor_msgs/PointField datatype enum -> numpy scalar type
_POINTFIELD_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cloud_to_xyz(msg) -> np.ndarray:
    """Extract an Nx3 float64 array from a PointCloud2 without copying fields."""

    fields = {f.name: f for f in msg.fields}
    missing = [axis for axis in ("x", "y", "z") if axis not in fields]
    if missing:
        raise ValueError(f"PointCloud2 is missing field(s): {', '.join(missing)}")

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [_POINTFIELD_DTYPES[fields[a].datatype] for a in ("x", "y", "z")],
            "offsets": [fields[a].offset for a in ("x", "y", "z")],
            "itemsize": msg.point_step,
        }
    )
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")

    count = len(msg.data) // msg.point_step
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=count)
    return np.column_stack((raw["x"], raw["y"], raw["z"])).astype("float64", copy=False)


class VoxelAccumulator:
    """Incremental voxel grid keeping the exact centroid of every occupied voxel.

    Memory scales with the number of occupied voxels rather than with the total
    number of points, so the whole run stays bounded even for millions of scans.
    """

    _SHIFT = 1 << 20  # keeps hashed indices non-negative for +/- 1M voxels

    def __init__(self, voxel_size_m: float):
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be > 0")
        self.voxel_size_m = float(voxel_size_m)
        self._keys = np.zeros(0, dtype="int64")
        self._sums = np.zeros((0, 3), dtype="float64")
        self._counts = np.zeros(0, dtype="int64")

    def _hash(self, points: np.ndarray) -> np.ndarray:
        idx = np.floor(points / self.voxel_size_m).astype("int64") + self._SHIFT
        if (idx < 0).any() or (idx >= (1 << 21)).any():
            raise ValueError("Point out of voxel-grid range; increase voxel size")
        return idx[:, 0] | (idx[:, 1] << 21) | (idx[:, 2] << 42)

    def add(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        keys = np.concatenate((self._keys, self._hash(points)))
        sums = np.vstack((self._sums, points))
        counts = np.concatenate((self._counts, np.ones(len(points), dtype="int64")))

        unique, inverse = np.unique(keys, return_inverse=True)
        merged_sums = np.zeros((len(unique), 3), dtype="float64")
        np.add.at(merged_sums, inverse, sums)
        merged_counts = np.bincount(inverse, weights=counts).astype("int64")

        self._keys, self._sums, self._counts = unique, merged_sums, merged_counts

    def points(self) -> np.ndarray:
        if len(self._keys) == 0:
            return np.zeros((0, 3), dtype="float64")
        return self._sums / self._counts[:, None]

    def __len__(self) -> int:
        return len(self._keys)


def read_cloud_index(db_path: Path, topic_name: str):
    """Return (message_ids, header-independent bag timestamps) for one topic."""

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            select messages.id, messages.timestamp
            from messages
            join topics on messages.topic_id = topics.id
            where topics.name = ?
            order by messages.timestamp
            """,
            (topic_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"No messages for topic '{topic_name}' in {db_path}")
    ids = np.array([r[0] for r in rows], dtype="int64")
    return ids


def nearest_pose_index(pose_times: np.ndarray, stamp: float, tolerance_s: float):
    """Index of the pose closest to ``stamp``, or None if outside tolerance."""

    i = int(np.searchsorted(pose_times, stamp))
    best, best_dt = None, np.inf
    for candidate in (i - 1, i):
        if 0 <= candidate < len(pose_times):
            dt = abs(pose_times[candidate] - stamp)
            if dt < best_dt:
                best, best_dt = candidate, dt
    return best if best_dt <= tolerance_s else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accumulate a global PCD map from recorded KISS-ICP poses."
    )
    parser.add_argument("--config", default="config/config.yaml", help="YAML config file")
    parser.add_argument("--bag", default=None, help="Original rosbag directory")
    parser.add_argument("--kiss-bag", default=None, help="Recorded /kiss/odometry bag")
    parser.add_argument("--topic", default=None, help="PointCloud2 topic")
    parser.add_argument("--out", default=None, help="Output .pcd (default: config output.map_pcd)")
    parser.add_argument("--stride", type=int, default=None, help="Use every Nth scan")
    parser.add_argument("--voxel", type=float, default=None, help="Map voxel size [m]")
    parser.add_argument("--min-range", type=float, default=None, help="Drop points closer than this")
    parser.add_argument("--max-range", type=float, default=None, help="Drop points beyond this")
    parser.add_argument(
        "--remove-ground", action="store_true", default=None, help="Drop points below --ground-z"
    )
    parser.add_argument("--ground-z", type=float, default=None, help="Ground threshold [m]")
    parser.add_argument(
        "--tolerance", type=float, default=None, help="Max stamp mismatch to accept [s]"
    )
    parser.add_argument(
        "--level",
        action="store_true",
        default=None,
        help="Gravity-align the map using the trajectory plane (same as pipeline_kiss --level)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    map_cfg = cfg.get("mapping", {})
    topic = args.topic or cfg["data"].get("pointcloud_topic", "/ouster/points")
    stride = args.stride if args.stride is not None else int(map_cfg.get("scan_stride", 10))
    voxel = args.voxel if args.voxel is not None else float(map_cfg.get("voxel_size_m", 0.25))
    min_range = args.min_range if args.min_range is not None else float(map_cfg.get("min_range_m", 1.0))
    max_range = args.max_range if args.max_range is not None else float(map_cfg.get("max_range_m", 50.0))
    remove_ground = args.remove_ground if args.remove_ground is not None else bool(map_cfg.get("remove_ground", False))
    ground_z = args.ground_z if args.ground_z is not None else float(map_cfg.get("ground_z_threshold_m", -1.5))
    tolerance = args.tolerance if args.tolerance is not None else float(map_cfg.get("pose_tolerance_s", 0.02))
    level_enabled = args.level if args.level is not None else bool(map_cfg.get("leveling_enabled", False))
    bag_dir = Path(args.bag or cfg["data"]["rosbag_dir"]).expanduser()
    kiss_bag = Path(args.kiss_bag or cfg["data"].get("kiss_bag", "data/kiss_output")).expanduser()
    out_pcd = Path(args.out or cfg["output"]["map_pcd"]).expanduser()
    for label, path in (("rosbag", bag_dir), ("KISS-ICP bag", kiss_bag)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    try:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        raise RuntimeError(
            "Reading point clouds needs a sourced ROS 2 environment. Run:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source ~/ros2_ws/install/setup.bash"
        ) from exc

    print("Building global point-cloud map from KISS-ICP poses")
    print(f"  rosbag     : {bag_dir}")
    print(f"  poses      : {kiss_bag}")
    print(f"  stride     : every {stride} scans")
    print(f"  voxel      : {voxel} m")
    print(f"  range gate : {min_range} .. {max_range} m")

    trajectory = sorted(
        load_kiss_odometry(kiss_bag, verbose=False),
        key=lambda step: step.timestamp_s,
    )
    backend_cfg = cfg.get("backend", {})
    local_poses = from_odometry_steps(
        trajectory,
        backend="kiss_icp",
        parent_frame=backend_cfg.get("parent_frame", "odom_lidar"),
        child_frame=backend_cfg.get("child_frame", "os_sensor"),
    )
    local_path = cfg["output"].get(
        "trajectory_local_csv", "outputs/trajectory_local.csv"
    )
    write_trajectory_csv(local_path, local_poses)
    pose_times = np.array([pose.timestamp for pose in local_poses], dtype="float64")
    transforms = [timed_pose_to_transform(pose) for pose in local_poses]
    order = np.argsort(pose_times, kind="stable")
    pose_times = pose_times[order]
    transforms = [transforms[i] for i in order]
    print(f"  loaded {len(trajectory)} poses "
          f"({pose_times[0]:.1f} .. {pose_times[-1]:.1f})")
    print(f"  local poses : {local_path}")

    if level_enabled:
        # Pre-multiplying the poses rotates positions and orientations together,
        # so the map stays consistent with a levelled trajectory.
        xyz = np.array([t[:3, 3] for t in transforms], dtype="float64")
        levelling, level_info = level_transform(xyz)
        transforms = [levelling @ t for t in transforms]
        print(f"  gravity alignment: {describe(level_info)}")

    db_path = find_bag_db(bag_dir)
    ids = read_cloud_index(db_path, topic)
    selected = ids[:: max(1, stride)]
    print(f"  {len(ids)} scans on {topic}, using {len(selected)}")

    grid = VoxelAccumulator(voxel)
    used = skipped = 0
    total_points = 0
    started = time.time()

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for n, message_id in enumerate(selected, start=1):
            row = conn.execute(
                "select data from messages where id = ?", (int(message_id),)
            ).fetchone()
            if row is None:
                continue

            msg = deserialize_message(bytes(row[0]), PointCloud2)
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

            pose_index = nearest_pose_index(pose_times, stamp, tolerance)
            if pose_index is None:
                skipped += 1
                continue

            points = cloud_to_xyz(msg)
            points = filter_points(
                points,
                min_range_m=min_range,
                max_range_m=max_range,
                remove_ground=remove_ground,
                ground_z_threshold_m=ground_z,
            )
            if len(points) == 0:
                continue

            transform = transforms[pose_index]
            world = points @ transform[:3, :3].T + transform[:3, 3]
            grid.add(world)

            used += 1
            total_points += len(points)
            if n == 1 or n % 25 == 0 or n == len(selected):
                elapsed = time.time() - started
                print(f"  {n}/{len(selected)} scans | {len(grid):,} voxels | {elapsed:.0f}s")

    if used == 0:
        raise RuntimeError(
            "No scan could be matched to a pose. Check --tolerance and that the "
            "odometry bag really comes from this rosbag."
        )

    cloud = grid.points()
    print(f"\nMerged {total_points:,} points from {used} scans "
          f"into {len(cloud):,} voxels")
    if skipped:
        print(f"  {skipped} scans had no pose within {tolerance}s "
              "(expected for scans recorded before /kiss/odometry capture started)")

    extent = cloud.max(axis=0) - cloud.min(axis=0)
    print(f"  map extent : {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m")

    out_pcd.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {out_pcd} ...")
    write_ascii_pcd(out_pcd, cloud)
    size_mb = out_pcd.stat().st_size / 1e6
    print(f"Wrote {out_pcd} ({len(cloud):,} points, {size_mb:.1f} MB)")
    if size_mb > 200:
        print("  tip: increase --voxel or lower --max-range for a smaller file")


if __name__ == "__main__":
    main()
