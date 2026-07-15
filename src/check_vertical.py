"""Decide what the odometry's vertical error really is, using GNSS altitude.

``level.py`` recovers the vertical by *assuming* the vehicle drives on a level
plane. That assumption is exactly what is in question here: the trajectory plane
says the odom frame is tilted 10 deg, while the Ouster IMU says the sensor sits
1.5 deg off vertical. Both cannot be right.

GNSS altitude settles it, because it is an *independent, unbiased* vertical
reference -- noisy (epv is typically 2-3x worse than eph), but not systematically
wrong. Instead of assuming flat ground we regress

    alt ~ a*x + b*y + c*z + d

over the odometry positions. Then:

  * (a, b, c) is the world vertical expressed in the odom frame
  * |(a, b, c)| ~ 1 means GNSS altitude and odometry z measure the same metre --
    the fit is meaningful and the implied tilt is trustworthy
  * |(a, b, c)| << 1 means they do not track each other at all: one of the two
    is essentially noise, and no rotation can reconcile them

    python3 -m src.check_vertical \\
        --kiss-bag ~/ros2_ws/data/kiss_output \\
        --gnss-csv /mnt/c/SensorSystems/xtrack_gnss_corrected/xtrack_global_position_t12.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .kiss_adapter import load_kiss_odometry
from .level import apply_rotation, level_rotation
from .odometry import trajectory_xyz


def load_gnss_columns(path: Path, columns: list[str]) -> dict:
    """Read arbitrary numeric columns from the corrected GNSS CSV."""

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("lat_lon_valid") not in {None, "1", "true", "True"}:
                continue
            try:
                rows.append([float(row[c]) for c in columns])
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    data = np.array(rows, dtype="float64")
    return {name: data[:, i] for i, name in enumerate(columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the odometry vertical against GNSS")
    parser.add_argument("--kiss-bag", required=True)
    parser.add_argument("--gnss-csv", required=True)
    parser.add_argument("--gnss-time-offset", type=float, default=-312.7144)
    parser.add_argument("--alt-column", default="alt", help="alt or alt_ellipsoid")
    parser.add_argument("--plot", default="outputs/plots/vertical_check.png")
    args = parser.parse_args()

    trajectory = load_kiss_odometry(Path(args.kiss_bag).expanduser(), verbose=False)
    xyz = trajectory_xyz(trajectory)
    times = np.array([s.timestamp_s for s in trajectory], dtype="float64")

    wanted = ["timestamp_sample", args.alt_column]
    gnss = load_gnss_columns(Path(args.gnss_csv).expanduser(), wanted + ["epv", "eph"])
    gnss_t = gnss["timestamp_sample"]
    order = np.argsort(gnss_t)
    gnss_t = gnss_t[order]
    gnss_alt = gnss[args.alt_column][order]
    epv = gnss["epv"][order]

    query = times + args.gnss_time_offset
    inside = (query >= gnss_t[0]) & (query <= gnss_t[-1])
    if inside.sum() < 100:
        raise RuntimeError("Too little overlap between odometry and GNSS")

    alt = np.interp(query[inside], gnss_t, gnss_alt)
    epv_i = np.interp(query[inside], gnss_t, epv)
    p = xyz[inside]

    print(f"{int(inside.sum())} poses overlap the GNSS record")
    print(f"\n[GNSS vertical quality]")
    print(f"  epv (vertical accuracy) : {epv_i.mean():.2f} m mean, "
          f"{epv_i.max():.2f} m max")
    print(f"  eph (horizontal)        : {gnss['eph'].mean():.2f} m mean")
    print(f"  altitude span           : {alt.max() - alt.min():.2f} m")
    step = np.abs(np.diff(alt))
    print(f"  sample-to-sample jitter : {np.median(step) * 100:.1f} cm median, "
          f"{step.max():.2f} m max")
    print(f"  -> a {alt.max() - alt.min():.1f} m span against {epv_i.mean():.1f} m "
          f"accuracy is {'mostly signal' if (alt.max()-alt.min()) > 4*epv_i.mean() else 'comparable to the noise'}")

    # --- regress the world vertical directly out of the odometry frame -----
    design = np.column_stack((p[:, 0], p[:, 1], p[:, 2], np.ones(len(p))))
    coefficients, *_ = np.linalg.lstsq(design, alt, rcond=None)
    predicted = design @ coefficients
    residual = alt - predicted
    up = coefficients[:3]
    scale = float(np.linalg.norm(up))
    total_variance = float(np.var(alt))
    r_squared = 1.0 - float(np.var(residual)) / total_variance if total_variance > 0 else 0.0

    print(f"\n[regression] alt ~ a*x + b*y + c*z + d")
    print(f"  (a, b, c) = ({coefficients[0]:+.4f}, {coefficients[1]:+.4f}, "
          f"{coefficients[2]:+.4f})")
    print(f"  |(a, b, c)| = {scale:.4f}   (1.0 = odometry z and GNSS alt agree in scale)")
    print(f"  R^2         = {r_squared:.4f}")
    print(f"  residual    = {np.sqrt(np.mean(residual**2)):.2f} m RMS "
          f"(GNSS epv is {epv_i.mean():.2f} m)")

    if scale < 0.3:
        print("\n  -> the odometry positions barely predict GNSS altitude at all.")
        print("     Either the GNSS altitude is dominated by noise, or the odometry")
        print("     vertical is unrelated to the real one. No rotation fixes this.")
    else:
        direction = up / scale
        tilt = np.degrees(np.arccos(np.clip(abs(direction[2]), 0.0, 1.0)))
        print(f"\n  -> world vertical in odom frame: "
              f"[{direction[0]:+.4f} {direction[1]:+.4f} {direction[2]:+.4f}]")
        print(f"     implied tilt: {tilt:.2f} deg")
        print(f"     compare: plane fit 10.01 deg, Ouster IMU 1.47 deg")

    # --- how do the candidate verticals compare? --------------------------
    rotation, plane_info = level_rotation(xyz)
    levelled = apply_rotation(xyz, rotation)[inside]

    print(f"\n[profiles over the overlapping window]")
    for name, series in (
        ("GNSS altitude", alt - alt[0]),
        ("odometry z (raw)", p[:, 2] - p[0, 2]),
        ("odometry z (plane-levelled)", levelled[:, 2] - levelled[0, 2]),
    ):
        correlation = float(np.corrcoef(series, alt)[0, 1])
        print(f"  {name:28s} span {series.max() - series.min():7.2f} m | "
              f"corr with GNSS alt {correlation:+.3f}")

    # --- plot -------------------------------------------------------------
    out = Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = times[inside] - times[inside][0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t0, alt - alt[0], label="GNSS altitude", linewidth=1.5)
    axes[0].fill_between(t0, alt - alt[0] - epv_i, alt - alt[0] + epv_i,
                         alpha=0.2, label="GNSS +/- epv")
    axes[0].plot(t0, p[:, 2] - p[0, 2], label="odometry z (raw)", linewidth=1.2)
    axes[0].plot(t0, levelled[:, 2] - levelled[0, 2],
                 label="odometry z (plane-levelled)", linewidth=1.2)
    if scale >= 0.3:
        axes[0].plot(t0, predicted - predicted[0], "--",
                     label="odometry projected on GNSS vertical", linewidth=1.2)
    axes[0].set_ylabel("height change [m]")
    axes[0].set_title("Vertical profile: odometry vs GNSS altitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(t0, residual, linewidth=1.0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("residual [m]")
    axes[1].set_title("GNSS altitude minus best-fit odometry vertical")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()