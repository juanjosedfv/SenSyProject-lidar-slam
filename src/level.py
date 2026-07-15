"""Level (gravity-align) the KISS-ICP odometry frame.

KISS-ICP has no IMU, so its ``odom_lidar`` frame inherits the sensor attitude of
the very first scan. If the LiDAR is mounted at an angle -- or the vehicle
happens to stand on a slope at t=0 -- the whole trajectory and map come out
tilted: driving on flat ground then shows up as a steady "climb".

Because the vehicle drives on (approximately) level ground, the best-fit plane
through the trajectory is a good estimate of the local horizontal. Its normal
therefore approximates the gravity direction, and rotating that normal onto +Z
levels both the trajectory and the map.

The residual of the plane fit tells you whether the assumption holds: a small
residual means the tilt really is a frame rotation, a large one means the
odometry has genuine vertical drift and this correction is not appropriate.

Diagnostics only:

    python3 -m src.level --kiss-bag ~/ros2_ws/data/kiss_output
"""

from __future__ import annotations

import numpy as np


def fit_plane(points: np.ndarray):
    """Least-squares fit of ``z = a*x + b*y + c``.

    Returns (a, b, c, residual_rms, residual_max).
    """

    points = np.asarray(points, dtype="float64")
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    residual = points[:, 2] - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.abs(residual).max()),
    )


def plane_normal(a: float, b: float) -> np.ndarray:
    """Upward unit normal of the plane ``z = a*x + b*y + c``."""

    normal = np.array([-a, -b, 1.0], dtype="float64")
    return normal / np.linalg.norm(normal)


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix R with ``R @ source == target`` (unit vectors).

    "Minimal" means it rotates about the axis perpendicular to both, so it adds
    no gratuitous spin about the target axis -- the map keeps its heading.
    """

    source = np.asarray(source, dtype="float64")
    target = np.asarray(target, dtype="float64")
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    axis = np.cross(source, target)
    cosine = float(np.dot(source, target))
    sine = float(np.linalg.norm(axis))

    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3, dtype="float64")
        # antiparallel: rotate 180 deg about any perpendicular axis
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, fallback)
        axis /= np.linalg.norm(axis)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * skew @ skew

    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype="float64",
    )
    return np.eye(3, dtype="float64") + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def level_rotation(points: np.ndarray):
    """Rotation that makes the best-fit plane of ``points`` horizontal.

    Returns (rotation_3x3, info dict with tilt_deg / residual_rms_m / ...).
    """

    a, b, c, residual_rms, residual_max = fit_plane(points)
    normal = plane_normal(a, b)
    rotation = rotation_between(normal, np.array([0.0, 0.0, 1.0]))

    info = {
        "plane_a": a,
        "plane_b": b,
        "plane_c": c,
        "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b)))),
        "tilt_azimuth_deg": float(np.degrees(np.arctan2(b, a))),
        "residual_rms_m": residual_rms,
        "residual_max_m": residual_max,
    }
    return rotation, info


def level_transform(points: np.ndarray):
    """4x4 homogeneous version of :func:`level_rotation` (no translation)."""

    rotation, info = level_rotation(points)
    transform = np.eye(4, dtype="float64")
    transform[:3, :3] = rotation
    return transform, info


def apply_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate an Nx3 array."""

    return np.asarray(points, dtype="float64") @ np.asarray(rotation).T


def gravity_from_rotation_axes(transforms, stride: int = 10, min_angle_deg: float = 1.0):
    """Estimate the gravity direction from the odometry *orientations*.

    A ground vehicle on flat terrain turns about the vertical axis only, so every
    incremental rotation expressed in the odom frame has (up to sign) the same
    axis: gravity. Averaging those axes therefore recovers the vertical without
    ever looking at the positions -- which makes it fully independent of
    :func:`level_rotation`, and a genuine cross-check rather than a restatement.

    Sign ambiguity (left turns give +axis, right turns -axis) is handled by
    taking the dominant eigenvector of the angle-weighted scatter matrix.

    Returns (unit_axis, info dict).
    """

    from scipy.spatial.transform import Rotation

    rotations = [np.asarray(t, dtype="float64")[:3, :3] for t in transforms]
    if len(rotations) <= stride:
        raise ValueError("Not enough poses for the requested stride")

    axes, weights = [], []
    for i in range(0, len(rotations) - stride, stride):
        delta = rotations[i + stride] @ rotations[i].T
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if np.degrees(angle) < min_angle_deg:
            continue  # too small to carry direction information
        axes.append(rotvec / angle)
        weights.append(angle)

    if len(axes) < 10:
        raise ValueError(
            f"Only {len(axes)} usable rotations above {min_angle_deg} deg; "
            "the trajectory may be too straight to estimate gravity this way"
        )

    axes_arr = np.asarray(axes, dtype="float64")
    weights_arr = np.asarray(weights, dtype="float64")
    scatter = (axes_arr * weights_arr[:, None]).T @ axes_arr
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, -1]
    if axis[2] < 0:
        axis = -axis

    alignment = np.abs(axes_arr @ axis)
    info = {
        "num_rotations": len(axes),
        "total_rotation_deg": float(np.degrees(weights_arr.sum())),
        "concentration": float(eigenvalues[-1] / eigenvalues.sum()),
        "mean_axis_deviation_deg": float(
            np.degrees(np.mean(np.arccos(np.clip(alignment, 0.0, 1.0))))
        ),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0)))),
    }
    return axis, info


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two vectors."""

    u = np.asarray(u, dtype="float64")
    v = np.asarray(v, dtype="float64")
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def describe(info: dict) -> str:
    """One-line summary suitable for logs and the report."""

    return (
        f"tilt {info['tilt_deg']:.2f} deg "
        f"(azimuth {info['tilt_azimuth_deg']:+.1f} deg), "
        f"plane-fit residual {info['residual_rms_m']:.2f} m RMS / "
        f"{info['residual_max_m']:.2f} m max"
    )


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Report the odometry frame tilt")
    parser.add_argument(
        "--kiss-bag", default="data/kiss_output", help="Recorded /kiss/odometry bag"
    )
    parser.add_argument(
        "--min-angle", type=float, default=1.0, help="Ignore turns below this [deg]"
    )
    args = parser.parse_args()

    if __package__ in {None, ""}:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from .kiss_adapter import load_kiss_odometry
    from .odometry import trajectory_xyz

    trajectory = load_kiss_odometry(Path(args.kiss_bag).expanduser(), verbose=False)
    xyz = trajectory_xyz(trajectory)
    rotation, info = level_rotation(xyz)
    levelled = apply_rotation(xyz, rotation)

    print(f"{len(xyz)} poses")
    print("\n[method 1] plane fit through the trajectory positions")
    print(f"  {describe(info)}")
    print(f"  z span before : {xyz[:, 2].max() - xyz[:, 2].min():6.2f} m")
    print(f"  z span after  : {levelled[:, 2].max() - levelled[:, 2].min():6.2f} m")
    print(f"  horizontal path before: "
          f"{np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum():.1f} m")
    print(f"  horizontal path after : "
          f"{np.linalg.norm(np.diff(levelled[:, :2], axis=0), axis=1).sum():.1f} m")
    if info["residual_rms_m"] > 3.0:
        print("  warning: large plane residual -- the tilt may be genuine vertical drift")

    plane_normal_vec = plane_normal(info["plane_a"], info["plane_b"])
    transforms = [step.transform for step in trajectory]

    print("\n[method 2] rotation axes of the odometry orientations (independent)")
    print("  Small turns carry little direction information, so the estimate is swept")
    print("  over increasing strides: convergence means noise, drift means bias.")
    print(f"\n  {'stride':>7} {'turns':>6} {'tot deg':>8} {'tilt':>7} "
          f"{'conc':>6} {'scatter':>8} {'vs plane':>9}")

    converged = None
    for stride in (10, 25, 50, 100, 200):
        try:
            axis, axis_info = gravity_from_rotation_axes(
                transforms, stride=stride, min_angle_deg=args.min_angle
            )
        except (ValueError, ImportError) as exc:
            print(f"  {stride:7d}  unavailable: {exc}")
            continue
        disagreement = angle_between(plane_normal_vec, axis)
        print(f"  {stride:7d} {axis_info['num_rotations']:6d} "
              f"{axis_info['total_rotation_deg']:8.0f} "
              f"{axis_info['tilt_deg']:6.2f}d {axis_info['concentration']:6.3f} "
              f"{axis_info['mean_axis_deviation_deg']:7.1f}d {disagreement:8.2f}d")
        converged = (axis, axis_info, disagreement)

    if converged is None:
        return

    axis, axis_info, disagreement = converged
    print(f"\n[cross-check] at the longest stride the two independent estimates give")
    print(f"  tilt magnitude : {info['tilt_deg']:.2f} deg (plane) vs "
          f"{axis_info['tilt_deg']:.2f} deg (rotation axes)")
    print(f"  full 3-D angle : {disagreement:.2f} deg apart")
    if disagreement < 5.0:
        print("  -> the two agree within the rotation-axis noise; the tilt is a frame")
        print("     rotation, not vertical drift")
    else:
        print("  -> directions still differ; the rotation-axis estimate is noisy, so")
        print("     judge mainly on the tilt magnitude and the plane-fit residual")


if __name__ == "__main__":
    main()