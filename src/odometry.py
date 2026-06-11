"""ICP odometry utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OdometryStep:
    timestamp_s: float
    source_file: str
    transform: "object"
    fitness: float
    inlier_rmse: float


def identity_transform():
    import numpy as np

    return np.eye(4, dtype="float64")


def register_icp(source, target, initial_transform, cfg: dict):
    """Register ``source`` into ``target`` with point-to-point ICP."""

    import numpy as np

    max_distance = float(cfg.get("icp_max_correspondence_distance_m", 1.5))
    max_iterations = int(cfg.get("icp_max_iterations", 50))
    tolerance = float(cfg.get("icp_tolerance_m", 1e-4))

    transform = np.array(initial_transform, dtype="float64", copy=True)
    previous_error = np.inf
    fitness = 0.0
    inlier_rmse = np.inf

    for _ in range(max_iterations):
        moved = transform_points(source, transform)
        distances, indices = nearest_neighbors(moved, target)
        mask = distances <= max_distance
        if mask.sum() < 3:
            break

        delta = best_fit_transform(moved[mask], target[indices[mask]])
        transform = delta @ transform
        inlier_rmse = float(np.sqrt(np.mean(distances[mask] ** 2)))
        fitness = float(mask.mean())

        if abs(previous_error - inlier_rmse) < tolerance:
            break
        previous_error = inlier_rmse

    return {
        "transformation": transform,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
    }


def nearest_neighbors(source, target):
    """Return nearest target distances and indices for each source point."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "ICP nearest-neighbor search requires SciPy. "
            "Run: pip install -r requirements.txt"
        ) from exc

    tree = cKDTree(target)
    return tree.query(source, k=1)


def transform_points(points, transform):
    """Apply a homogeneous transform to an ``Nx3`` point array."""

    import numpy as np

    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def best_fit_transform(source, target):
    """Return rigid transform mapping ``source`` points onto ``target`` points."""

    import numpy as np

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_zero = source - source_centroid
    target_zero = target - target_centroid
    covariance = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    transform = identity_transform()
    transform[:3, :3] = rotation
    transform[:3, 3] = target_centroid - rotation @ source_centroid
    return transform


def transform_cloud(cloud, transform):
    """Return transformed points."""

    return transform_points(cloud, transform)


def write_ascii_pcd(path, points):
    """Write a simple XYZ ASCII PCD file."""

    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    with output.open("w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def run_icp_odometry(scans: list[tuple[float, str, object]], cfg: dict):
    """Estimate a trajectory and accumulated map from timestamped point arrays."""

    import numpy as np

    from .preprocess import voxel_downsample

    if not scans:
        raise ValueError("No scans available for odometry")

    use_scan_to_map = bool(cfg.get("use_scan_to_map", False))
    map_voxel_size = float(cfg.get("map_voxel_size_m", 0.35))

    trajectory = [
        OdometryStep(
            timestamp_s=scans[0][0],
            source_file=scans[0][1],
            transform=identity_transform(),
            fitness=1.0,
            inlier_rmse=0.0,
        )
    ]
    accumulated_map = transform_cloud(scans[0][2], trajectory[0].transform)
    previous_cloud = scans[0][2]
    previous_transform = trajectory[0].transform

    for timestamp_s, source_file, current_cloud in scans[1:]:
        if use_scan_to_map:
            target = accumulated_map
            initial_guess = previous_transform
            result = register_icp(current_cloud, target, initial_guess, cfg)
            current_transform = np.asarray(result["transformation"])
        else:
            result = register_icp(current_cloud, previous_cloud, identity_transform(), cfg)
            current_to_previous = np.asarray(result["transformation"])
            current_transform = previous_transform @ current_to_previous

        trajectory.append(
            OdometryStep(
                timestamp_s=timestamp_s,
                source_file=source_file,
                transform=current_transform,
                fitness=float(result["fitness"]),
                inlier_rmse=float(result["inlier_rmse"]),
            )
        )

        accumulated_map = np.vstack(
            (accumulated_map, transform_cloud(current_cloud, current_transform))
        )
        if map_voxel_size > 0:
            accumulated_map = voxel_downsample(accumulated_map, map_voxel_size)

        previous_cloud = current_cloud
        previous_transform = current_transform

    return trajectory, accumulated_map


def trajectory_xyz(trajectory: list[OdometryStep]):
    """Extract translation columns from a trajectory as ``Nx3``."""

    import numpy as np

    return np.array([step.transform[:3, 3] for step in trajectory], dtype="float64")
