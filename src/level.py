"""Level the KISS-ICP odometry frame by removing the LiDAR mount rotation.

KISS-ICP has no absolute attitude reference, so its ``odom_lidar`` frame is
simply the sensor frame of the very first scan. The Ouster on the XTrack is
mounted at an angle, so that frame is tilted with respect to gravity and the
whole trajectory and map come out tilted with it: driving on level ground shows
up as a steady "climb" of z_span = path_length * tan(tilt).

Two independent estimators of the vertical are provided, and they differ in
what they assume:

  [plane]  Fit a plane through the trajectory POSITIONS and take its normal.
           Assumes the ground is level. Cheap and robust in magnitude, but its
           AZIMUTH is only as well conditioned as the trajectory's aspect
           ratio: on an out-and-back down a corridor (45 x 241 m here, 5.3:1)
           the across-track coefficient is ~5x noisier than the along-track
           one, so the tilt direction is poorly determined.

  [axes]   Take the rotation axes of the odometry ORIENTATIONS. A ground
           vehicle turns about the vertical, so every incremental rotation
           expressed in the odom frame shares one axis: gravity. This never
           looks at the positions and never assumes level ground. Its premise
           -- that the platform turns about gravity -- is independently
           confirmed by /fmu/out/vehicle_attitude: median tilt 1.73 deg over
           the full record, never above 5.88 deg. See probe_attitude.py.

On this dataset the two agree on MAGNITUDE to 0.31 deg (10.01 vs 10.32) and
disagree on DIRECTION by 6.09 deg. That is not a conflict: [axes] has a
direction standard error of scatter/sqrt(N) ~= 2-3.5 deg, so the ~5 deg
residual is [plane]'s azimuth error, exactly as the 5.3:1 aspect ratio
predicts. **[axes] is the better direction; [plane] is the better-tested
default.** Hence `method=` rather than a silent switch.

The plane-fit residual is the drift test: it is 0.97 m RMS against a 42.43 m
z span (2.3 %), i.e. the tilt is a constant frame rotation and not accumulated
vertical drift. A large residual would mean the opposite and would make this
whole correction inappropriate.

Diagnostics -- runs both estimators and compares them:

    python3 -m src.level --kiss-bag ~/ros2_ws/data/kiss_output
"""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


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
    rotation = rotation_between(normal, UP)

    info = {
        "method": "plane",
        "plane_a": a,
        "plane_b": b,
        "plane_c": c,
        "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b)))),
        "tilt_azimuth_deg": float(np.degrees(np.arctan2(b, a))),
        "residual_rms_m": residual_rms,
        "residual_max_m": residual_max,
    }
    return rotation, info


def axis_level_rotation(
    transforms, stride: int = 50, min_angle_deg: float = 1.0, points=None
):
    """Rotation that levels the frame using the ORIENTATION-derived vertical.

    Same contract as :func:`level_rotation`, but the vertical comes from
    :func:`gravity_from_rotation_axes` instead of a plane through the
    positions. Assumes only that the platform turns about gravity -- not that
    the ground is level -- and its direction is better conditioned when the
    trajectory is long and thin.

    If ``points`` is given, the same ``residual_rms_m`` / ``residual_max_m``
    keys as :func:`level_rotation` are reported, so the two info dicts stay
    interchangeable for callers that log them. The meaning differs and that is
    the point: for [plane] the residual is the least-squares minimum by
    construction, whereas here it is the spread of the positions about a plane
    perpendicular to the INDEPENDENTLY measured gravity. It is therefore an
    honest estimate of real terrain relief plus vertical drift, and it will be
    larger than [plane]'s. Larger is not worse.

    Returns (rotation_3x3, info dict).
    """

    axis, info = gravity_from_rotation_axes(
        transforms, stride=stride, min_angle_deg=min_angle_deg
    )
    rotation = rotation_between(axis, UP)

    info = dict(info)
    info["method"] = "axes"
    info["stride"] = stride
    # Match level_rotation()'s convention: it reports arctan2(b, a) for the
    # plane z = a*x + b*y + c, i.e. the azimuth of the DOWN-SLOPE gradient,
    # which is 180 deg from the upward normal. `axis` here is the upward
    # gravity direction, so negate before taking the azimuth. Getting this
    # wrong makes two estimates that are 31.7 deg apart print as 212 deg apart.
    info["tilt_azimuth_deg"] = float(np.degrees(np.arctan2(-axis[1], -axis[0])))
    # scatter/sqrt(N): how well the mean direction itself is pinned down, as
    # opposed to how scattered the individual axes are.
    info["direction_stderr_deg"] = float(
        info["mean_axis_deviation_deg"] / np.sqrt(max(info["num_rotations"], 1))
    )

    if points is not None:
        levelled = apply_rotation(points, rotation)
        residual = levelled[:, 2] - float(np.mean(levelled[:, 2]))
        info["residual_rms_m"] = float(np.sqrt(np.mean(residual**2)))
        info["residual_max_m"] = float(np.abs(residual).max())

    return rotation, info


def level_transform(
    points: np.ndarray,
    transforms=None,
    method: str = "plane",
    stride: int = 50,
    min_angle_deg: float = 1.0,
):
    """4x4 homogeneous levelling transform (rotation only, no translation).

    ``method="plane"`` reproduces the original behaviour exactly and is the
    default so that existing results do not move underneath anyone; it is also
    the A/B control. ``method="axes"`` uses the orientation-derived vertical
    and needs ``transforms`` (the per-pose 4x4s).
    """

    if method == "plane":
        rotation, info = level_rotation(points)
    elif method == "axes":
        if transforms is None:
            raise ValueError("method='axes' needs the per-pose transforms")
        rotation, info = axis_level_rotation(
            transforms, stride, min_angle_deg, points=points
        )
    else:
        raise ValueError(f"unknown method {method!r}; use 'plane' or 'axes'")

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

    if info.get("method") == "axes":
        return (
            f"tilt {info['tilt_deg']:.2f} deg "
            f"(azimuth {info['tilt_azimuth_deg']:+.1f} deg), "
            f"from {info['num_rotations']} turns / "
            f"{info['total_rotation_deg']:.0f} deg total, "
            f"direction SE {info['direction_stderr_deg']:.2f} deg"
        )
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
    parser.add_argument(
        "--axes-stride",
        type=int,
        default=50,
        help="Stride for the [axes] estimator in the head-to-head (default 50: "
        "the sweep below shows it has the lowest direction SE)",
    )
    args = parser.parse_args()

    if __package__ in {None, ""}:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from kiss_icp.adapter import load_kiss_odometry
    from slam_icp.trajectory import trajectory_xyz

    trajectory = load_kiss_odometry(Path(args.kiss_bag).expanduser(), verbose=False)
    xyz = trajectory_xyz(trajectory)
    transforms = [step.transform for step in trajectory]

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

    residual_fraction = info["residual_rms_m"] / max(
        xyz[:, 2].max() - xyz[:, 2].min(), 1e-9
    )
    print(f"\n  [drift test] residual {info['residual_rms_m']:.2f} m RMS against a "
          f"{xyz[:, 2].max() - xyz[:, 2].min():.2f} m z span = "
          f"{100 * residual_fraction:.1f} %")
    if info["residual_rms_m"] > 3.0:
        print("  -> LARGE residual: the tilt may be genuine vertical drift, and")
        print("     rotating it away would be inappropriate.")
    else:
        print("  -> small residual: the tilt is a constant FRAME ROTATION, not")
        print("     accumulated vertical drift. Rotating it away is the right fix.")

    plane_normal_vec = plane_normal(info["plane_a"], info["plane_b"])

    print("\n[method 2] rotation axes of the odometry orientations (independent)")
    print("  Small turns carry little direction information, so the estimate is swept")
    print("  over increasing strides: convergence means noise, drift means bias.")
    print("  dir SE = scatter/sqrt(N): how well the MEAN direction is pinned down.")
    print(f"\n  {'stride':>7} {'turns':>6} {'tot deg':>8} {'tilt':>7} "
          f"{'conc':>6} {'scatter':>8} {'dir SE':>7} {'vs plane':>9}")

    sweep = []
    for stride in (10, 25, 50, 100, 200):
        try:
            axis, axis_info = gravity_from_rotation_axes(
                transforms, stride=stride, min_angle_deg=args.min_angle
            )
        except (ValueError, ImportError) as exc:
            print(f"  {stride:7d}  unavailable: {exc}")
            continue
        disagreement = angle_between(plane_normal_vec, axis)
        se = axis_info["mean_axis_deviation_deg"] / np.sqrt(axis_info["num_rotations"])
        print(f"  {stride:7d} {axis_info['num_rotations']:6d} "
              f"{axis_info['total_rotation_deg']:8.0f} "
              f"{axis_info['tilt_deg']:6.2f}d {axis_info['concentration']:6.3f} "
              f"{axis_info['mean_axis_deviation_deg']:7.1f}d {se:6.2f}d "
              f"{disagreement:8.2f}d")
        sweep.append((stride, axis, axis_info, disagreement, se))

    if not sweep:
        return

    tilts = [s[2]["tilt_deg"] for s in sweep]
    print(f"\n  tilt across strides: {np.mean(tilts):.2f} +/- {np.std(tilts):.2f} deg "
          f"(range {max(tilts) - min(tilts):.2f})")
    if np.std(tilts) < 1.0:
        print("  -> converged, no drift with stride: the magnitude is real.")
    else:
        print("  -> still moving with stride: treat the magnitude with caution.")

    # ---- head to head -------------------------------------------------------
    print(f"\n[head to head] plane normal vs rotation axis (stride {args.axes_stride})")
    try:
        rot_axes, info_axes = axis_level_rotation(
            transforms, stride=args.axes_stride, min_angle_deg=args.min_angle,
            points=xyz,
        )
    except (ValueError, ImportError) as exc:
        print(f"  unavailable: {exc}")
        return
    levelled_axes = apply_rotation(xyz, rot_axes)

    print(f"  [plane] {describe(info)}")
    print(f"  [axes ] {describe(info_axes)}")
    print(f"\n  magnitude : {info['tilt_deg']:.2f} vs {info_axes['tilt_deg']:.2f} deg "
          f"-> {abs(info['tilt_deg'] - info_axes['tilt_deg']):.2f} deg apart")
    az_gap = abs((info["tilt_azimuth_deg"] - info_axes["tilt_azimuth_deg"] + 180)
                 % 360 - 180)
    sep = angle_between(plane_normal_vec, rot_axes.T @ UP)
    print(f"  azimuth   : {az_gap:.1f} deg apart (both in the down-slope "
          f"convention)")
    print(f"  direction : {sep:.2f} deg apart in 3-D")
    print(f"              [axes] direction SE is "
          f"{info_axes['direction_stderr_deg']:.2f} deg, so roughly "
          f"{np.sqrt(max(sep**2 - info_axes['direction_stderr_deg']**2, 0)):.2f} deg")
    print( "              of that is [plane]'s azimuth error -- which is what a "
           "long, thin")
    print( "              trajectory predicts, since the across-track plane "
           "coefficient is")
    print( "              the poorly conditioned one.")

    print(f"\n  {'':22} {'[plane]':>10} {'[axes]':>10}")
    for name, a, b in (
        ("z span after [m]",
         levelled[:, 2].max() - levelled[:, 2].min(),
         levelled_axes[:, 2].max() - levelled_axes[:, 2].min()),
        ("horiz. path after [m]",
         np.linalg.norm(np.diff(levelled[:, :2], axis=0), axis=1).sum(),
         np.linalg.norm(np.diff(levelled_axes[:, :2], axis=0), axis=1).sum()),
    ):
        print(f"  {name:22} {a:10.2f} {b:10.2f}")

    print("\n  NOTE: do NOT pick a method on 'z span after'. The plane fit is the")
    print("  least-squares minimiser of exactly that quantity, so it wins by")
    print("  construction -- comparing on it is circular. The only fair test is")
    print("  horizontal RMSE against GNSS, which lives in pipeline_kiss.py:")
    print("    level_transform(xyz, transforms=transforms, method='axes')")


if __name__ == "__main__":
    main()
