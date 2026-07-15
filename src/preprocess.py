from __future__ import annotations
import numpy as np



def filter_points(
    points,
    *,
    min_range_m: float = 1.0,
    max_range_m: float = 80.0,
    remove_ground: bool = False,
    ground_z_threshold_m: float = -1.5,
):
    if points.size == 0:
        return points

    finite = np.isfinite(points).all(axis=1)
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = finite & (ranges >= min_range_m) & (ranges <= max_range_m)
    if remove_ground:
        mask &= points[:, 2] > ground_z_threshold_m
    return points[mask]


def voxel_downsample(points, voxel_size_m: float = 0.25):
    if points.size == 0 or voxel_size_m <= 0:
        return points

    voxel_index = np.floor(points / voxel_size_m).astype("int64")
    _, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    downsampled = np.zeros((counts.size, 3), dtype="float64")
    np.add.at(downsampled, inverse, points)
    downsampled /= counts[:, None]
    return downsampled


def preprocess_scan(points, cfg: dict):
    filtered = filter_points(
        points,
        min_range_m=float(cfg.get("min_range_m", 1.0)),
        max_range_m=float(cfg.get("max_range_m", 80.0)),
        remove_ground=bool(cfg.get("remove_ground", False)),
        ground_z_threshold_m=float(cfg.get("ground_z_threshold_m", -1.5)),
    )
    return voxel_downsample(
        filtered,
        float(cfg.get("voxel_size_m", 0.25)),
    )
