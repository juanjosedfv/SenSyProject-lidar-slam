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

from .dataset import resolve_dataset_paths
from .evaluation import (
    align_2d_rigid,
    compute_rmse,
    gnss_to_enu,
    interpolate_ground_truth,
    load_gnss_ground_truth,
    origin_from_gnss,
    save_reference_diagnostic_plot,
    save_evaluation_plots,
)
from .geo import enu_to_latlon, enu_to_ned
from .load_laz import discover_laz_files, load_scan, select_files
from .odometry import run_icp_odometry, trajectory_xyz, write_ascii_pcd
from .preprocess import preprocess_scan


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_trajectory_outputs(trajectory, xyz, aligned_xy, origin, cfg):
    timestamps = np.array([step.timestamp_s for step in trajectory], dtype="float64")
    up = xyz[:, 2]
    lat, lon, alt = enu_to_latlon(aligned_xy[:, 0], aligned_xy[:, 1], up, origin)

    est = {
        "timestamp": timestamps,
        "source_file": [step.source_file for step in trajectory],
        "x_lidar_m": xyz[:, 0],
        "y_lidar_m": xyz[:, 1],
        "z_lidar_m": xyz[:, 2],
        "east_m": aligned_xy[:, 0],
        "north_m": aligned_xy[:, 1],
        "up_m": up,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "icp_fitness": [step.fitness for step in trajectory],
        "icp_inlier_rmse": [step.inlier_rmse for step in trajectory],
        "icp_degenerate": [step.degenerate for step in trajectory],
    }

    trajectory_columns = list(est.keys())
    with Path(cfg["output"]["trajectory_csv"]).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(trajectory_columns)
        for i in range(len(timestamps)):
            writer.writerow([est[col][i] for col in trajectory_columns])

    east_vel = np.gradient(est["east_m"], timestamps)
    north_vel = np.gradient(est["north_m"], timestamps)
    up_vel = np.gradient(est["up_m"], timestamps)
    vn, ve, vd = enu_to_ned(east_vel, north_vel, up_vel)
    with Path(cfg["output"]["velocity_csv"]).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "vn_mps", "ve_mps", "vd_mps"])
        writer.writerows(zip(timestamps, vn, ve, vd))

    return est


def _path_length_xy(xy):
    if len(xy) < 2:
        return 0.0
    deltas = np.diff(xy[:, :2], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _latlon_to_relative_xy(lat, lon):
    radius_m = 6_378_137.0
    lat0_rad = np.radians(lat[0])
    east = np.radians(lon - lon[0]) * radius_m * np.cos(lat0_rad)
    north = np.radians(lat - lat[0]) * radius_m
    return np.column_stack((east, north))


def _load_plotjuggler_raw_gps_xy(path: Path, timestamps):
    lat_col = "/fmu/out/vehicle_gps_position/latitude_deg"
    lon_col = "/fmu/out/vehicle_gps_position/longitude_deg"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((float(row["bag_time"]), float(row[lat_col]), float(row[lon_col])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        return None

    data = np.array(rows, dtype="float64")
    ts = np.asarray(timestamps, dtype="float64")
    lat = np.interp(ts, data[:, 0], data[:, 1])
    lon = np.interp(ts, data[:, 0], data[:, 2])
    return _latlon_to_relative_xy(lat, lon)


def _load_fmu_local_xy(path: Path, timestamps):
    x_col = "/fmu/out/vehicle_local_position_v1/x"
    y_col = "/fmu/out/vehicle_local_position_v1/y"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((float(row["bag_time"]), float(row[x_col]), float(row[y_col])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        return None

    data = np.array(rows, dtype="float64")
    ts = np.asarray(timestamps, dtype="float64")
    north = np.interp(ts, data[:, 0], data[:, 1])
    east = np.interp(ts, data[:, 0], data[:, 2])
    east -= east[0]
    north -= north[0]
    return np.column_stack((east, north))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIDAR-based positioning pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--max-scans", type=int, default=None, help="Override config max_scans")
    parser.add_argument("--scan-stride", type=int, default=None, help="Override config scan_stride")
    parser.add_argument("--start-index", type=int, default=None, help="Override config start_index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["plot_dir"]).mkdir(parents=True, exist_ok=True)

    dataset = resolve_dataset_paths(cfg)
    files = discover_laz_files(dataset.pointcloud_dir, cfg["data"].get("pointcloud_glob", "*.laz"))
    max_scans = args.max_scans if args.max_scans is not None else cfg["data"].get("max_scans")
    scan_stride = args.scan_stride if args.scan_stride is not None else int(cfg["data"].get("scan_stride", 1))
    start_index = args.start_index if args.start_index is not None else int(cfg["data"].get("start_index", 0))
    selected_files = select_files(
        files,
        max_scans=max_scans,
        scan_stride=scan_stride,
        start_index=start_index,
    )
    if len(selected_files) < 2:
        raise RuntimeError("Need at least two point clouds for ICP odometry")

    print("LIDAR-based positioning pipeline")
    print(f"Dataset: {cfg['data']['dataset_name']}")
    print(f"ROS bag: {dataset.rosbag_db}")
    print(f"Point cloud topic: {dataset.pointcloud_topic}")
    if dataset.rosbag_pointcloud_count is not None:
        print(f"ROS bag point cloud messages: {dataset.rosbag_pointcloud_count}")
    print(f"Exported LAZ scans: {len(files)} from {dataset.pointcloud_dir}")
    print(f"Corrected GNSS: {dataset.gnss_csv}")
    print(f"Using {len(selected_files)} scans for this run")
    print(f"Selection: start_index={start_index}, scan_stride={scan_stride}")

    scans = []
    for index, path in enumerate(selected_files, start=1):
        scan = load_scan(path)
        cloud = preprocess_scan(scan.points, cfg["preprocessing"])
        scans.append((scan.timestamp_s, str(scan.path), cloud))
        if index == 1 or index % 25 == 0 or index == len(selected_files):
            print(f"Loaded/preprocessed {index}/{len(selected_files)} scans")

    scan_timestamps = [s[0] for s in scans]
    gnss_raw = None
    gnss_enu = None
    reference = None
    gnss_initial_poses = None
    use_gnss_seed = bool(cfg["odometry"].get("use_gnss_seed", False))
    gnss_time_offset_s = float(cfg["data"].get("gnss_time_offset_s", 0.0))
    if dataset.gnss_csv and Path(dataset.gnss_csv).exists():
        gnss_raw = load_gnss_ground_truth(dataset.gnss_csv)
        origin = origin_from_gnss(gnss_raw)
        gnss_enu = gnss_to_enu(gnss_raw, origin)
        reference = interpolate_ground_truth(
            gnss_enu,
            scan_timestamps,
            time_offset_s=gnss_time_offset_s,
        )
        valid_mask = reference["valid"].astype(bool, copy=False)
        if use_gnss_seed and valid_mask.all():
            gnss_initial_poses = [
                {"east_m": reference["east_m"][i], "north_m": reference["north_m"][i], "up_m": reference["up_m"][i]}
                for i in range(len(scans))
            ]
            print("Seeding ICP with GNSS initial poses")
        elif use_gnss_seed:
            print(f"Warning: only {valid_mask.sum()}/{len(scans)} GNSS matches, not using as seed")
        else:
            print("GNSS is evaluation-only; running LiDAR odometry independently")

    trajectory, accumulated_map = run_icp_odometry(
        scans, cfg["odometry"], initial_poses=gnss_initial_poses,
    )
    write_ascii_pcd(cfg["output"]["map_pcd"], accumulated_map)
    xyz = trajectory_xyz(trajectory)

    if gnss_raw is None:
        gnss_raw = load_gnss_ground_truth(dataset.gnss_csv)
        origin = origin_from_gnss(gnss_raw)
        gnss = gnss_to_enu(gnss_raw, origin)
        reference = interpolate_ground_truth(
            gnss,
            [step.timestamp_s for step in trajectory],
            time_offset_s=gnss_time_offset_s,
        )
    else:
        gnss = gnss_enu
    valid = reference["valid"].astype(bool, copy=False)
    if valid.sum() < 2:
        raise RuntimeError("Estimated scan timestamps do not overlap with GNSS ground truth")

    reference_xy = np.column_stack((reference["east_m"], reference["north_m"]))
    raw_lidar_path_m = _path_length_xy(xyz[:, :2])
    gnss_path_m = _path_length_xy(reference_xy[valid])
    raw_lidar_displacement_m = float(
        np.linalg.norm(xyz[valid, :2][-1] - xyz[valid, :2][0])
    )
    gnss_displacement_m = float(
        np.linalg.norm(reference_xy[valid][-1] - reference_xy[valid][0])
    )
    aligned_valid, rotation, translation = align_2d_rigid(
        xyz[valid, :2],
        reference_xy[valid],
    )
    aligned_xy = xyz[:, :2] @ rotation.T + translation
    aligned_lidar_path_m = _path_length_xy(aligned_xy[valid])
    rmse_m = compute_rmse(aligned_valid - reference_xy[valid])
    path_length_ratio = raw_lidar_path_m / gnss_path_m if gnss_path_m > 0 else None
    displacement_ratio = (
        raw_lidar_displacement_m / gnss_displacement_m
        if gnss_displacement_m > 0
        else None
    )

    est = _write_trajectory_outputs(trajectory, xyz, aligned_xy, origin, cfg)
    save_evaluation_plots(
        {"timestamp": est["timestamp"][valid]},
        aligned_valid,
        {
            "east_m": reference["east_m"][valid],
            "north_m": reference["north_m"][valid],
        },
        cfg["output"]["plot_dir"],
    )

    finite_icp_rmse = [
        step.inlier_rmse for step in trajectory if np.isfinite(step.inlier_rmse)
    ]
    metrics = {
        "num_scans": len(trajectory),
        "num_gnss_matches": int(valid.sum()),
        "gnss_used_for_odometry": gnss_initial_poses is not None,
        "gnss_time_offset_s": gnss_time_offset_s,
        "horizontal_rmse_m": rmse_m,
        "raw_lidar_path_m": raw_lidar_path_m,
        "aligned_lidar_path_m": aligned_lidar_path_m,
        "gnss_path_m": gnss_path_m,
        "raw_lidar_to_gnss_path_ratio": path_length_ratio,
        "raw_lidar_displacement_m": raw_lidar_displacement_m,
        "gnss_displacement_m": gnss_displacement_m,
        "raw_lidar_to_gnss_displacement_ratio": displacement_ratio,
        "mean_icp_fitness": sum(step.fitness for step in trajectory) / len(trajectory),
        "mean_icp_inlier_rmse_m": (
            float(np.mean(finite_icp_rmse)) if finite_icp_rmse else None
        ),
        "num_degenerate_icp_frames": sum(1 for step in trajectory if step.degenerate),
    }

    diagnostic_series = {
        "raw LIDAR ICP": xyz[:, :2],
        "corrected GNSS": reference_xy,
    }

    fmu_path = Path("data/rosbag/rosbag_data/plotjuggler_fmu.csv")
    if fmu_path.exists():
        raw_gps_xy = _load_plotjuggler_raw_gps_xy(
            fmu_path, [step.timestamp_s for step in trajectory]
        )
        if raw_gps_xy is not None:
            metrics["plotjuggler_raw_gps_path_m"] = _path_length_xy(raw_gps_xy)
            diagnostic_series["raw GPS topic"] = raw_gps_xy

        fmu_xy = _load_fmu_local_xy(fmu_path, [step.timestamp_s for step in trajectory])
        if fmu_xy is not None:
            metrics["fmu_local_path_m"] = _path_length_xy(fmu_xy)
            diagnostic_series["PX4 local position"] = fmu_xy

    save_reference_diagnostic_plot(diagnostic_series, cfg["output"]["plot_dir"])

    metrics_path = Path(cfg["output"].get("metrics_json", output_dir / "metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote trajectory: {cfg['output']['trajectory_csv']}")
    print(f"Wrote velocity: {cfg['output']['velocity_csv']}")
    print(f"Wrote map: {cfg['output']['map_pcd']}")
    print(f"Horizontal RMSE vs corrected GNSS: {rmse_m:.2f} m")
    print(f"Raw LIDAR path length: {raw_lidar_path_m:.2f} m")
    print(f"Corrected GNSS path length: {gnss_path_m:.2f} m")
    if displacement_ratio is not None and not 0.5 <= displacement_ratio <= 2.0:
        print(
            "Warning: LiDAR and GNSS endpoint displacement disagree strongly. "
            "Verify timestamps, sensor frames, and ICP convergence."
        )
    if metrics["num_degenerate_icp_frames"]:
        print(
            f"Warning: {metrics['num_degenerate_icp_frames']}/{len(trajectory)} scans had "
            "degenerate ICP matches (see icp_degenerate column in the trajectory CSV)."
        )


if __name__ == "__main__":
    main()
