"""Point cloud loading utilities for exported Ouster LAZ scans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class PointCloudScan:
    """A single timestamped point cloud scan."""

    path: Path
    timestamp_ns: int
    timestamp_s: float
    points: "object"


def discover_laz_files(pointcloud_dir: str | Path, glob_pattern: str = "*.laz") -> list[Path]:
    """Return sorted LAZ files from a directory."""

    directory = Path(pointcloud_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Point cloud directory does not exist: {directory}")
    return sorted(directory.glob(glob_pattern))


def timestamp_ns_from_path(path: str | Path) -> int:
    """Extract the nanosecond timestamp from names like cloud_1780....laz."""

    match = re.search(r"(\d{16,})", Path(path).stem)
    if not match:
        raise ValueError(f"Could not extract timestamp from point cloud file: {path}")
    return int(match.group(1))


def select_files(
    files: Iterable[Path],
    *,
    max_scans: int | None = None,
    scan_stride: int = 1,
    start_index: int = 0,
) -> list[Path]:
    """Apply deterministic sub-sampling to keep experiments quick."""

    if scan_stride < 1:
        raise ValueError("scan_stride must be >= 1")
    selected = list(files)[start_index::scan_stride]
    if max_scans is not None and max_scans > 0:
        selected = selected[:max_scans]
    return selected


def read_laz_points(path: str | Path):
    """Load XYZ coordinates from a LAZ/LAS file as an ``Nx3`` numpy array."""

    try:
        import laspy
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "LAZ loading requires laspy with lazrs support. In this project use: "
            "conda activate sensor_system && python -m pip install 'laspy[lazrs]'"
        ) from exc

    las = laspy.read(path)
    return np.column_stack((las.x, las.y, las.z)).astype("float64", copy=False)


def load_scan(path: str | Path) -> PointCloudScan:
    """Load one timestamped LAZ scan."""

    scan_path = Path(path)
    timestamp_ns = timestamp_ns_from_path(scan_path)
    return PointCloudScan(
        path=scan_path,
        timestamp_ns=timestamp_ns,
        timestamp_s=timestamp_ns / 1e9,
        points=read_laz_points(scan_path),
    )
