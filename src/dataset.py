"""Dataset path resolution for the provided ROS bag and corrected GNSS files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class DatasetPaths:
    rosbag_dir: Path
    rosbag_db: Path
    rosbag_metadata: Path
    pointcloud_dir: Path
    gnss_csv: Path
    pointcloud_topic: str
    rosbag_pointcloud_count: int | None


def resolve_dataset_paths(cfg: dict) -> DatasetPaths:
    """Resolve and validate the project data paths from ``config.yaml``."""

    data_cfg = cfg["data"]
    rosbag_dir = Path(data_cfg.get("rosbag_dir", "data/rosbag"))
    rosbag_db = Path(data_cfg.get("rosbag_db", rosbag_dir / "rosbag_0.db3"))
    rosbag_metadata = Path(data_cfg.get("rosbag_metadata", rosbag_dir / "metadata.yaml"))
    pointcloud_dir = Path(
        data_cfg.get("pointcloud_dir", rosbag_dir / "rosbag_data" / "laz_clouds")
    )
    gnss_csv = Path(
        data_cfg.get(
            "gnss_csv",
            Path("data/xtrack_gnss_corrected/xtrack_global_position_t12.csv"),
        )
    )
    pointcloud_topic = data_cfg.get("pointcloud_topic", "/ouster/points")

    required = {
        "rosbag_dir": rosbag_dir,
        "rosbag_db": rosbag_db,
        "rosbag_metadata": rosbag_metadata,
        "pointcloud_dir": pointcloud_dir,
        "gnss_csv": gnss_csv,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset paths:\n" + "\n".join(missing))

    return DatasetPaths(
        rosbag_dir=rosbag_dir,
        rosbag_db=rosbag_db,
        rosbag_metadata=rosbag_metadata,
        pointcloud_dir=pointcloud_dir,
        gnss_csv=gnss_csv,
        pointcloud_topic=pointcloud_topic,
        rosbag_pointcloud_count=rosbag_topic_message_count(rosbag_db, pointcloud_topic),
    )


def rosbag_topic_message_count(rosbag_db: Path, topic_name: str) -> int | None:
    """Return the stored message count for a topic in a ROS 2 sqlite bag."""

    with sqlite3.connect(rosbag_db) as conn:
        row = conn.execute(
            """
            select count(*)
            from messages
            join topics on messages.topic_id = topics.id
            where topics.name = ?
            """,
            (topic_name,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0])
