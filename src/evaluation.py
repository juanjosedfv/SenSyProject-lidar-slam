from __future__ import annotations
import csv
import os
from pathlib import Path
import numpy as np
from .geo import GeoOrigin, latlon_to_enu
import matplotlib.pyplot as plt


def load_gnss_ground_truth(path: str | Path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if "lat_lon_valid" in row and row["lat_lon_valid"] not in {"1", "true", "True"}:
                continue
            try:
                rows.append(
                    {
                        "timestamp_sample": float(row["timestamp_sample"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "alt": float(row.get("alt") or 0.0),
                    }
                )
            except (TypeError, ValueError):
                continue

    if not rows:
        raise ValueError(f"No valid GNSS rows found in {path}")

    return {
        key: np.array([row[key] for row in rows], dtype="float64")
        for key in ["timestamp_sample", "lat", "lon", "alt"]
    }


def origin_from_gnss(df) -> GeoOrigin:
    return GeoOrigin(
        lat_deg=float(df["lat"][0]),
        lon_deg=float(df["lon"][0]),
        alt_m=float(df["alt"][0]) if "alt" in df else 0.0,
    )


def gnss_to_enu(df, origin: GeoOrigin):
    alt = df["alt"] if "alt" in df else 0.0
    east, north, up = latlon_to_enu(df["lat"], df["lon"], alt, origin)
    return {**df, "east_m": east, "north_m": north, "up_m": up}


def interpolate_ground_truth(gnss_df, timestamps_s, time_offset_s: float = 0.0):
    time = gnss_df["timestamp_sample"].astype("float64", copy=False)
    order = np.argsort(time)
    time = time[order]
    timestamps = np.asarray(timestamps_s, dtype="float64")
    query_time = timestamps + float(time_offset_s)
    mask = (query_time >= time[0]) & (query_time <= time[-1])

    data = {"timestamp": timestamps, "valid": mask}
    for col in ["east_m", "north_m", "up_m"]:
        values = gnss_df[col].astype("float64", copy=False)[order]
        data[col] = np.interp(query_time, time, values)
    return data


def align_2d_rigid(estimated_xy, reference_xy):
    est = np.asarray(estimated_xy, dtype="float64")
    ref = np.asarray(reference_xy, dtype="float64")
    est_center = est.mean(axis=0)
    ref_center = ref.mean(axis=0)
    est_zero = est - est_center
    ref_zero = ref - ref_center

    cov = est_zero.T @ ref_zero
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        rotation = np.eye(2, dtype="float64")
        translation = ref_center - est_center
        aligned = est + translation
        return aligned, rotation, translation
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = ref_center - est_center @ rotation.T
    aligned = est @ rotation.T + translation
    return aligned, rotation, translation


def compute_rmse(error_xy):
    err = np.asarray(error_xy, dtype="float64")
    horizontal = np.linalg.norm(err[:, :2], axis=1)
    return float(np.sqrt(np.mean(horizontal**2)))


def save_evaluation_plots(estimated_df, aligned_xy, reference_df, output_dir: str | Path):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    ref_xy = np.column_stack((reference_df["east_m"], reference_df["north_m"]))
    error = np.linalg.norm(aligned_xy - ref_xy, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], label="GNSS ground truth", linewidth=2)
    ax.plot(aligned_xy[:, 0], aligned_xy[:, 1], label="LIDAR odometry aligned", linewidth=2)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Estimated trajectory vs corrected GNSS")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "trajectory_vs_gnss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(estimated_df["timestamp"], error)
    ax.set_xlabel("Timestamp [s]")
    ax.set_ylabel("Horizontal error [m]")
    ax.set_title("Aligned trajectory error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "horizontal_error.png", dpi=180)
    plt.close(fig)


def save_reference_diagnostic_plot(series_by_name: dict, output_dir: str | Path):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, xy in series_by_name.items():
        points = np.asarray(xy, dtype="float64")
        if len(points) < 2:
            continue
        relative = points[:, :2] - points[0, :2]
        ax.plot(relative[:, 0], relative[:, 1], marker=".", markersize=3, label=name)

    ax.set_xlabel("Relative East / local x [m]")
    ax.set_ylabel("Relative North / local y [m]")
    ax.set_title("Relative motion diagnostics")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "reference_diagnostics.png", dpi=180)
    plt.close(fig)
