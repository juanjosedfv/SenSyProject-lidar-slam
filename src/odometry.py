from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_CKD_TREE = None


@dataclass(frozen=True)
class OdometryStep:
    timestamp_s: float
    source_file: str
    transform: "object"
    fitness: float
    inlier_rmse: float
    degenerate: bool = False


def identity_transform():
    return np.eye(4, dtype="float64")


def orthonormalize_rotation(transform):
    transform = np.array(transform, dtype="float64", copy=True)
    u, _, vt = np.linalg.svd(transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transform[:3, :3] = rotation
    return transform


def invert_rigid_transform(transform):
    inv = np.eye(4, dtype="float64")
    transform = orthonormalize_rotation(transform)
    rotation = transform[:3, :3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ transform[:3, 3]
    return inv


def register_icp(source, target, initial_transform, cfg: dict):
    max_distance = float(cfg.get("icp_max_correspondence_distance_m", 1.5))
    max_iterations = int(cfg.get("icp_max_iterations", 50))
    tolerance = float(cfg.get("icp_tolerance_m", 1e-4))
    max_correction_m = float(cfg.get("icp_max_correction_m", 0.0))
    degenerate_rmse_m = float(cfg.get("icp_degenerate_rmse_m", 1e-6))

    transform = np.array(initial_transform, dtype="float64", copy=True)
    initial_translation = transform[:3, 3].copy()
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
        if max_correction_m > 0:
            correction = transform[:3, 3] - initial_translation
            correction_norm = np.linalg.norm(correction)
            if correction_norm > max_correction_m:
                transform[:3, 3] = initial_translation + correction * (max_correction_m / correction_norm)

        inlier_rmse = float(np.sqrt(np.mean(distances[mask] ** 2)))
        fitness = float(mask.mean())

        if abs(previous_error - inlier_rmse) < tolerance:
            break
        previous_error = inlier_rmse

    return {
        "transformation": transform,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "degenerate": inlier_rmse < degenerate_rmse_m,
    }


def nearest_neighbors(source, target):
    global _CKD_TREE
    if _CKD_TREE is None:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError(
                "Run: pip install -r requirements.txt"
            ) from exc
        _CKD_TREE = cKDTree

    target = np.asarray(target, dtype="float64")
    t_mask = np.isfinite(target).all(axis=1)
    if not t_mask.all():
        target = target[t_mask]
    source = np.asarray(source, dtype="float64")
    s_mask = np.isfinite(source).all(axis=1)
    if not s_mask.all():
        source = np.where(s_mask[:, None], source, 1e6)

    tree = _CKD_TREE(target)
    dist, idx = tree.query(source, k=1)
    return dist, idx


def transform_points(points, transform):
    points = np.asarray(points, dtype="float64")
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def best_fit_transform(source, target):
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


def run_icp_odometry(
    scans: list[tuple[float, str, object]],
    cfg: dict,
    initial_poses: list[dict] | None = None,
):

    from .preprocess import voxel_downsample

    if not scans:
        raise ValueError("No scans available for odometry")

    use_scan_to_map = bool(cfg.get("use_scan_to_map", False))
    use_motion_model = bool(cfg.get("use_motion_model", True))
    map_voxel_size = float(cfg.get("map_voxel_size_m", 0.35))
    progress_interval = max(1, int(cfg.get("progress_interval", 25)))

    print("Initializing ICP nearest-neighbor backend")
    nearest_neighbors(
        np.zeros((1, 3), dtype="float64"),
        np.zeros((1, 3), dtype="float64"),
    )
    print("Running ICP odometry")

    gnss_transforms = None
    if initial_poses is not None:
        if len(initial_poses) != len(scans):
            raise ValueError(
                "initial_poses must contain exactly one pose per scan "
                f"({len(initial_poses)} poses for {len(scans)} scans)"
            )
        gnss_transforms = []
        for pose in initial_poses:
            T = identity_transform()
            T[0, 3] = float(pose.get("east_m", 0.0))
            T[1, 3] = float(pose.get("north_m", 0.0))
            T[2, 3] = float(pose.get("up_m", 0.0))
            gnss_transforms.append(T)
        first_pose_inv = invert_rigid_transform(gnss_transforms[0])
        gnss_transforms = [first_pose_inv @ T for T in gnss_transforms]

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
    previous_relative = identity_transform()
    degenerate_frames = 0

    for i, (timestamp_s, source_file, current_cloud) in enumerate(scans[1:], start=1):
        if gnss_transforms is not None:
            if use_scan_to_map:
                initial_guess = gnss_transforms[i]
            else:
                initial_guess = (
                    invert_rigid_transform(gnss_transforms[i - 1])
                    @ gnss_transforms[i]
                )
        elif use_scan_to_map:
            if use_motion_model:
                initial_guess = previous_relative @ previous_transform
            else:
                initial_guess = previous_transform
        else:
            initial_guess = previous_relative if use_motion_model else identity_transform()
        bootstrap_count = int(cfg.get("bootstrap_frames", 3))
        if gnss_transforms is not None and use_scan_to_map and i <= bootstrap_count:
            current_transform = gnss_transforms[i].copy()
            result = {
                "transformation": current_transform,
                "fitness": 0.0,
                "inlier_rmse": np.nan,
            }
            current_relative = current_transform @ invert_rigid_transform(previous_transform)
            if i == bootstrap_count:
                print(f"  ICP bootstrap complete after {bootstrap_count} frames, "
                      "switching to scan-to-map refinement")
        elif use_scan_to_map:
            target = accumulated_map
            result = register_icp(current_cloud, target, initial_guess, cfg)
            current_transform = np.asarray(result["transformation"])
            current_relative = current_transform @ invert_rigid_transform(previous_transform)
        else:
            result = register_icp(current_cloud, previous_cloud, initial_guess, cfg)
            current_to_previous = np.asarray(result["transformation"])
            current_transform = previous_transform @ current_to_previous
            current_relative = current_to_previous

        is_degenerate = bool(result.get("degenerate", False)) or not np.isfinite(
            result["inlier_rmse"]
        )
        if is_degenerate:
            degenerate_frames += 1
            current_relative = previous_relative
            current_transform = (
                current_relative @ previous_transform
                if use_scan_to_map
                else previous_transform @ current_relative
            )
            print(
                f"  Warning: degenerate/failed ICP match at scan {i + 1}/{len(scans)} "
                f"(inlier_rmse={result['inlier_rmse']:.2e}, fitness={result['fitness']:.3f}); "
                "holding previous motion estimate instead of trusting it"
            )

        current_transform = orthonormalize_rotation(current_transform)
        previous_relative = orthonormalize_rotation(np.asarray(current_relative, dtype="float64"))

        trajectory.append(
            OdometryStep(
                timestamp_s=timestamp_s,
                source_file=source_file,
                transform=current_transform,
                fitness=float(result["fitness"]),
                inlier_rmse=float(result["inlier_rmse"]),
                degenerate=is_degenerate,
            )
        )

        accumulated_map = np.vstack(
            (accumulated_map, transform_cloud(current_cloud, current_transform))
        )
        if map_voxel_size > 0:
            accumulated_map = voxel_downsample(accumulated_map, map_voxel_size)

        previous_cloud = current_cloud
        previous_transform = current_transform

        completed = i + 1
        if completed == 2 or completed % progress_interval == 0 or completed == len(scans):
            print(
                f"ICP odometry {completed}/{len(scans)} scans "
                f"(fitness={result['fitness']:.3f}, "
                f"RMSE={result['inlier_rmse']:.3f} m)"
            )

    if degenerate_frames:
        print(
            f"ICP flagged {degenerate_frames}/{len(scans) - 1} scans as degenerate "
            "matches (held previous motion estimate instead of trusting them)"
        )

    return trajectory, accumulated_map


def trajectory_xyz(trajectory: list[OdometryStep]):
    return np.array([step.transform[:3, 3] for step in trajectory], dtype="float64")
