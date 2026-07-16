"""KISS-ICP pipeline: recorded /kiss/odometry -> project deliverables.

Same evaluation stack as ``main.py`` (geo.py / evaluation.py, unchanged), but the
front-end is the recorded KISS-ICP odometry instead of LAZ + hand-written ICP.
No LAZ files required.

Clock offset convention follows ``evaluation.interpolate_ground_truth``:

    t_gnss = t_lidar + gnss_time_offset_s

so the configured -312.7144 s means the LiDAR clock runs ~313 s ahead of the
PX4/GNSS clock. That value was calibrated on the LAZ timestamps; use
``--scan-offset`` to verify it also holds for the bag's header stamps.

Run from the repository root with ROS 2 sourced:

    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 -m src.pipeline_kiss --kiss-bag ~/ros2_ws/data/kiss_output
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .evaluation import (
    align_2d_rigid,
    compute_rmse,
    gnss_to_enu,
    interpolate_ground_truth,
    load_gnss_ground_truth,
    origin_from_gnss,
    save_evaluation_plots,
    save_reference_diagnostic_plot,
)
from .geo import enu_to_latlon, enu_to_ned
from .kiss_adapter import load_kiss_odometry
from .level import apply_rotation, describe, level_rotation
from .odometry import trajectory_xyz


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_length_xy(xy) -> float:
    """Total 2D path length of an Nx2/Nx3 polyline."""

    points = np.asarray(xy, dtype="float64")
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def evaluate_offset(gnss, xyz, timestamps, offset: float):
    """Align the odometry to GNSS for one clock offset and score it."""

    reference = interpolate_ground_truth(gnss, timestamps, time_offset_s=offset)
    valid = reference["valid"].astype(bool, copy=False)
    if valid.sum() < 2:
        return None

    reference_xy = np.column_stack((reference["east_m"], reference["north_m"]))
    aligned_valid, rotation, translation = align_2d_rigid(
        xyz[valid, :2], reference_xy[valid]
    )
    errors = np.linalg.norm(aligned_valid - reference_xy[valid], axis=1)
    return {
        "offset": float(offset),
        "reference": reference,
        "valid": valid,
        "reference_xy": reference_xy,
        "aligned_valid": aligned_valid,
        "aligned_xy": xyz[:, :2] @ rotation.T + translation,
        "errors": errors,
        "rmse": compute_rmse(aligned_valid - reference_xy[valid]),
    }


def scan_offsets(gnss, xyz, timestamps, centre: float, span: float, step: float):
    """Brute-force the clock offset that minimises the aligned RMSE."""

    print(f"\nScanning clock offset {centre - span:+.1f} .. {centre + span:+.1f} s "
          f"(step {step:g} s)")
    print(f"  {'offset [s]':>12} {'matched':>9} {'RMSE [m]':>10}")
    results = []
    for offset in np.arange(centre - span, centre + span + 0.5 * step, step):
        scored = evaluate_offset(gnss, xyz, timestamps, float(offset))
        if scored is None:
            print(f"  {offset:12.2f} {'0':>9} {'-':>10}")
            continue
        results.append(scored)
        print(f"  {offset:12.2f} {int(scored['valid'].sum()):9d} {scored['rmse']:10.2f}")

    if not results:
        raise RuntimeError("No offset in the scan range overlaps the GNSS data")
    best = min(results, key=lambda r: r["rmse"])
    print(f"  -> best offset {best['offset']:+.2f} s, RMSE {best['rmse']:.2f} m")
    return best


def write_trajectory_csv(path, trajectory, xyz, aligned_xy, origin) -> dict:
    """Write the per-pose trajectory CSV and return the assembled columns."""

    timestamps = np.array([step.timestamp_s for step in trajectory], dtype="float64")
    up = xyz[:, 2]
    lat, lon, alt = enu_to_latlon(aligned_xy[:, 0], aligned_xy[:, 1], up, origin)

    columns = {
        "timestamp": timestamps,
        "x_lidar_m": xyz[:, 0],
        "y_lidar_m": xyz[:, 1],
        "z_lidar_m": xyz[:, 2],
        "east_m": aligned_xy[:, 0],
        "north_m": aligned_xy[:, 1],
        "up_m": up,
        "lat": lat,
        "lon": lon,
        "alt": alt,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))
        for i in range(len(timestamps)):
            writer.writerow([columns[name][i] for name in columns])
    return columns


def write_velocity_csv(path, timestamps, east_m, north_m, up_m) -> None:
    """Differentiate the aligned trajectory and write NED velocities."""

    east_vel = np.gradient(east_m, timestamps)
    north_vel = np.gradient(north_m, timestamps)
    up_vel = np.gradient(up_m, timestamps)
    vn, ve, vd = enu_to_ned(east_vel, north_vel, up_vel)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "vn_mps", "ve_mps", "vd_mps"])
        writer.writerows(zip(timestamps, vn, ve, vd))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recorded KISS-ICP odometry against corrected GNSS."
    )
    parser.add_argument("--config", default="config/config.yaml", help="YAML config file")
    parser.add_argument(
        "--kiss-bag",
        default=None,
        help="Recorded odometry bag directory (default: config data.kiss_bag)",
    )
    parser.add_argument(
        "--gnss-csv", default=None, help="Corrected GNSS CSV (default: config data.gnss_csv)"
    )
    parser.add_argument(
        "--odom-topic", default="/kiss/odometry", help="Recorded odometry topic"
    )
    parser.add_argument(
        "--gnss-time-offset",
        type=float,
        default=None,
        help="Override config data.gnss_time_offset_s (t_gnss = t_lidar + offset)",
    )
    parser.add_argument(
        "--level",
        action="store_true",
                help=(
            "Compensate the LiDAR mount rotation by fitting a plane through the "
            "trajectory. KISS-ICP has no absolute attitude reference, so a tilted "
            "sensor mount tilts the whole frame and shrinks the horizontal "
            "projection by cos(tilt). Measured at 10.0 deg here; see README."
        ),
    )
    parser.add_argument(
        "--scan-offset",
        action="store_true",
        help="Brute-force the best clock offset instead of trusting the configured one",
    )
    parser.add_argument("--scan-span", type=float, default=60.0, help="Scan +/- this many s")
    parser.add_argument("--scan-step", type=float, default=5.0, help="Scan step in s")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_cfg = cfg["output"]
    Path(out_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(out_cfg["plot_dir"]).mkdir(parents=True, exist_ok=True)

    kiss_bag = Path(
        args.kiss_bag or cfg["data"].get("kiss_bag", "data/kiss_output")
    ).expanduser()
    gnss_csv = Path(args.gnss_csv or cfg["data"]["gnss_csv"]).expanduser()
    for label, path in (("KISS-ICP bag", kiss_bag), ("GNSS CSV", gnss_csv)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    offset = args.gnss_time_offset
    if offset is None:
        offset = float(cfg["data"].get("gnss_time_offset_s", 0.0) or 0.0)

    print("LiDAR positioning pipeline (KISS-ICP odometry)")
    print(f"  odometry bag : {kiss_bag}")
    print(f"  GNSS truth   : {gnss_csv}")

    # --- estimate ---------------------------------------------------------
    trajectory = load_kiss_odometry(kiss_bag, topic_name=args.odom_topic)
    xyz = trajectory_xyz(trajectory)
    timestamps = np.array([step.timestamp_s for step in trajectory], dtype="float64")

    level_info = None
    if args.level:
        rotation, level_info = level_rotation(xyz)
        before_z = xyz[:, 2].max() - xyz[:, 2].min()
        xyz = apply_rotation(xyz, rotation)
        print(f"\nGravity alignment: {describe(level_info)}")
        print(f"  z span {before_z:.1f} m -> {xyz[:, 2].max() - xyz[:, 2].min():.1f} m")
        if level_info["residual_rms_m"] > 3.0:
            print("  warning: large plane residual, tilt may be genuine vertical drift")

    # --- ground truth -----------------------------------------------------
    gnss_raw = load_gnss_ground_truth(gnss_csv)
    origin = origin_from_gnss(gnss_raw)
    gnss = gnss_to_enu(gnss_raw, origin)
    gnss_t = gnss_raw["timestamp_sample"]

    print(f"\nGNSS: {len(gnss_raw['lat'])} fixes, origin "
          f"({origin.lat_deg:.6f}, {origin.lon_deg:.6f})")
    print(f"  GNSS stamps      : {gnss_t.min():.1f} .. {gnss_t.max():.1f}")
    print(f"  odometry stamps  : {timestamps[0]:.1f} .. {timestamps[-1]:.1f}")
    print(f"  clock offset     : {offset:+.4f} s  (t_gnss = t_lidar + offset)")
    print(f"  -> GNSS queried at {timestamps[0] + offset:.1f} .. "
          f"{timestamps[-1] + offset:.1f}")

    # --- align & score ----------------------------------------------------
    if args.scan_offset:
        best = scan_offsets(gnss, xyz, timestamps, offset, args.scan_span, args.scan_step)
        offset = best["offset"]
        scored = best
    else:
        scored = evaluate_offset(gnss, xyz, timestamps, offset)
        if scored is None:
            raise RuntimeError(
                "Odometry timestamps do not overlap the GNSS ground truth with offset "
                f"{offset:+.4f} s. Re-run with --scan-offset to search for a better one."
            )

    valid = scored["valid"]
    reference = scored["reference"]
    reference_xy = scored["reference_xy"]
    aligned_valid = scored["aligned_valid"]
    aligned_xy = scored["aligned_xy"]
    errors = scored["errors"]
    rmse_m = scored["rmse"]
    print(f"\n  matched poses: {int(valid.sum())}/{len(trajectory)}")

    raw_path_m = path_length_xy(xyz)
    gnss_path_m = path_length_xy(reference_xy[valid])
    ratio = raw_path_m / gnss_path_m if gnss_path_m > 0 else None

    raw_disp_m = float(np.linalg.norm(xyz[valid, :2][-1] - xyz[valid, :2][0]))
    gnss_disp_m = float(np.linalg.norm(reference_xy[valid][-1] - reference_xy[valid][0]))
    disp_ratio = raw_disp_m / gnss_disp_m if gnss_disp_m > 0 else None

    # --- deliverables -----------------------------------------------------
    est = write_trajectory_csv(
        out_cfg["trajectory_csv"], trajectory, xyz, aligned_xy, origin
    )
    write_velocity_csv(
        out_cfg["velocity_csv"], est["timestamp"], est["east_m"], est["north_m"], est["up_m"]
    )
    save_evaluation_plots(
        {"timestamp": est["timestamp"][valid]},
        aligned_valid,
        {"east_m": reference["east_m"][valid], "north_m": reference["north_m"][valid]},
        out_cfg["plot_dir"],
    )
    save_reference_diagnostic_plot(
        {"KISS-ICP (raw odom)": xyz[:, :2], "corrected GNSS": reference_xy},
        out_cfg["plot_dir"],
    )

    metrics = {
        "source": "kiss_icp_ros",
        "odometry_bag": str(kiss_bag),
        "odometry_topic": args.odom_topic,
        "gnss_time_offset_s": offset,
        "gnss_offset_scanned": bool(args.scan_offset),
        "gravity_levelled": bool(args.level),
        "frame_tilt_deg": level_info["tilt_deg"] if level_info else None,
        "frame_tilt_residual_rms_m": level_info["residual_rms_m"] if level_info else None,
        "num_poses": len(trajectory),
        "num_gnss_matches": int(valid.sum()),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "horizontal_rmse_m": rmse_m,
        "horizontal_error_mean_m": float(np.mean(errors)),
        "horizontal_error_median_m": float(np.median(errors)),
        "horizontal_error_p95_m": float(np.percentile(errors, 95)),
        "horizontal_error_max_m": float(np.max(errors)),
        "raw_lidar_path_m": raw_path_m,
        "gnss_path_m": gnss_path_m,
        "raw_lidar_to_gnss_path_ratio": ratio,
        "raw_lidar_displacement_m": raw_disp_m,
        "gnss_displacement_m": gnss_disp_m,
        "raw_lidar_to_gnss_displacement_ratio": disp_ratio,
    }
    metrics_path = Path(out_cfg.get("metrics_json", "outputs/metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- report -----------------------------------------------------------
    print()
    print(f"Wrote trajectory : {out_cfg['trajectory_csv']}")
    print(f"Wrote velocity   : {out_cfg['velocity_csv']}")
    print(f"Wrote metrics    : {metrics_path}")
    print(f"Wrote plots      : {out_cfg['plot_dir']}")
    print()
    print(f"Horizontal RMSE vs corrected GNSS : {rmse_m:.2f} m")
    print(f"  median {np.median(errors):.2f} m | p95 {np.percentile(errors, 95):.2f} m "
          f"| max {np.max(errors):.2f} m")
    print(f"LiDAR path {raw_path_m:.1f} m vs GNSS path {gnss_path_m:.1f} m"
          + (f" (ratio {ratio:.3f})" if ratio else ""))
    if ratio is not None and not 0.9 <= ratio <= 1.1:
        print("  note: path lengths differ by >10%, check odometry scale/drift")
    if disp_ratio is not None and not 0.5 <= disp_ratio <= 2.0:
        print("  warning: endpoint displacements disagree strongly; "
              "verify the clock offset (try --scan-offset)")


if __name__ == "__main__":
    main()