#!/usr/bin/env python3
"""Is the platform actually tilted, or is the odometry drifting?

The Pixhawk publishes a fused attitude solution at 100 Hz on
/fmu/out/vehicle_attitude (px4_msgs/msg/VehicleAttitude, 81799 messages here).
EKF2 has already fused accelerometer + gyro + magnetometer, so this is a
gravity reference, ready-made, over the whole record -- not one static window,
and with no complementary filter, CDR decoding or aspn_msgs required.

It settles the open question in the project's Findings: the trajectory plane
fit says 10.01 deg, the dominant rotation axis says 10.82 deg, the Ouster IMU
at rest says 1.47 deg. Those disagree.

    python3 -m src.probe_attitude --bag ~/ros2_ws/data/rosbag

Runs in the ROS 2 environment. px4_msgs is already built -- it is what makes
/fmu/out/vehicle_gps_position readable. Nothing is pip-installed.

CONVENTIONS -- the two easy ways to get this silently wrong:

  * PX4 orders its quaternion [w, x, y, z]. ROS orders it [x, y, z, w].
    Reading PX4's q as if it were ROS's gives a plausible-looking but wrong
    attitude, with no error anywhere.
  * PX4's q rotates body (FRD: Forward-Right-Down) -> local NED. Both z axes
    point DOWN, so the angle between the platform's down axis and gravity is
        tilt = arccos(R[2,2]),   R[2,2] = 1 - 2*(x^2 + y^2)
    which equals arccos(cos(roll)*cos(pitch)).

CAVEAT, and it matters: this is the PIXHAWK's attitude, not the Ouster's. The
two are rigidly attached but their mount rotation is unknown and unmodelled
here, so "the vehicle is level" is not literally "the LiDAR is level". What a
constant mount rotation CANNOT explain is time-varying tilt -- see the plot.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def quat_to_rpy_tilt(q_wxyz):
    """PX4 [w,x,y,z], body(FRD)->NED. Returns roll, pitch, yaw, tilt in rad.

    tilt = angle between the platform's down axis and gravity, i.e. how far
    from level it is, independent of heading.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    r22 = 1.0 - 2.0 * (x * x + y * y)          # body down . NED down
    tilt = np.arccos(np.clip(r22, -1.0, 1.0))
    return roll, pitch, yaw, tilt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True)
    p.add_argument("--topic", default="/fmu/out/vehicle_attitude")
    p.add_argument("--storage", default="sqlite3", help="or 'mcap'")
    p.add_argument("--plot", default="outputs/plots/pixhawk_attitude.png")
    args = p.parse_args(argv)

    import rosbag2_py
    from px4_msgs.msg import VehicleAttitude
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=args.storage),
                rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    ts, tss, qs, resets = [], [], [], []
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        m = deserialize_message(data, VehicleAttitude)
        ts.append(m.timestamp)
        tss.append(m.timestamp_sample)
        qs.append(list(m.q))
        resets.append(m.quat_reset_counter)

    if not qs:
        raise SystemExit(f"no messages on {args.topic}")

    q = np.asarray(qs, dtype=np.float64)          # PX4 order: w, x, y, z
    t_us = np.asarray(ts, dtype=np.float64)
    t = (t_us - t_us[0]) / 1e6                    # PX4 timestamps are MICROseconds
    rst = np.asarray(resets, dtype=np.int64)

    print(f"{args.topic}: {len(q)} messages")
    print(f"  duration : {t[-1]:.1f} s  ->  {len(q)/max(t[-1],1e-9):.1f} Hz")
    print(f"  timestamp        raw first : {ts[0]}")
    print(f"  timestamp_sample raw first : {tss[0]}")
    print( "    (PX4 clock, microseconds. The -312.7144 s offset relates this to")
    print( "     the LiDAR clock -- and is NOT needed for anything below.)")
    nq = np.linalg.norm(q, axis=1)
    print(f"  |q| : min {nq.min():.6f}  max {nq.max():.6f}   (must be ~1)")
    nres = int((np.diff(rst) != 0).sum())
    print(f"  EKF2 quaternion resets : {nres}"
          + ("" if nres == 0 else "   <- yaw jumps there; roll/pitch usually survive"))

    roll, pitch, yaw, tilt = quat_to_rpy_tilt(q)
    d = np.degrees

    print("\nattitude over the whole record [deg]:")
    print(f"  {'':8s} {'median':>9} {'mean':>9} {'p5':>9} {'p95':>9} "
          f"{'min':>9} {'max':>9}")
    for name, v in (("roll", d(roll)), ("pitch", d(pitch)), ("TILT", d(tilt))):
        print(f"  {name:8s} {np.median(v):>9.2f} {v.mean():>9.2f} "
              f"{np.percentile(v, 5):>9.2f} {np.percentile(v, 95):>9.2f} "
              f"{v.min():>9.2f} {v.max():>9.2f}")

    td = d(tilt)
    print("\nfraction of the record spent beyond a given tilt:")
    for c in (1, 2, 3, 5, 10):
        print(f"  > {c:2d} deg : {100*(td > c).mean():5.1f} %")

    med = float(np.median(td))
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Pixhawk EKF2, 100 Hz, full record : median tilt {med:.2f} deg")
    print( "  Ouster IMU, gravity at rest       :  1.47 deg   (inertial)")
    print( "  trajectory plane fit              : 10.01 deg   (from odometry)")
    print( "  dominant rotation axis            : 10.82 deg   (from odometry)")
    if med < 4.0:
        print("\n  -> The platform is essentially LEVEL, per a fused 100 Hz")
        print("     solution across the entire route. It agrees with the Ouster")
        print("     IMU and contradicts both geometric estimates.")
        print("     Both INERTIAL sensors say level. Both estimators derived")
        print("     FROM the odometry say 10 deg. The odometry is the outlier:")
        print("     the 42 m climb is drift -- not terrain, not a mount angle.")
        print("\n     Fourth independent line of evidence, alongside the")
        print("     return-to-start closure (comes back within ~1.5 m in XY, so")
        print("     the true dz is ~0), the GNSS altitude, and the Ouster IMU.")
    elif med > 8.0:
        print(f"\n  -> The platform really IS tilted ~{med:.0f} deg, which")
        print("     contradicts the Ouster IMU's 1.47 deg. That would need")
        print("     explaining -- most likely the Pixhawk->Ouster mount")
        print("     rotation. Do NOT use this as a gravity prior until it is")
        print("     resolved.")
    else:
        print(f"\n  -> Intermediate ({med:.2f} deg). Neither story is clean;")
        print("     read the time series before concluding anything.")

    print("\n  CAVEAT: this is the PIXHAWK's attitude. The Pixhawk->Ouster mount")
    print("  rotation is unknown, so a CONSTANT offset from the LiDAR frame is")
    print("  possible. A constant mount cannot explain time-varying tilt.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        a1.plot(t, d(roll), lw=0.6, label="roll")
        a1.plot(t, d(pitch), lw=0.6, label="pitch")
        a1.axhline(0, color="0.6", lw=0.8)
        a1.set_ylabel("[deg]")
        a1.legend(loc="upper right")
        a1.grid(alpha=0.3)
        a1.set_title("Pixhawk EKF2 attitude, 100 Hz (was in the bag all along)")
        a2.plot(t, td, lw=0.6, color="tab:purple", label="tilt from level")
        a2.axhline(10.01, color="tab:red", ls="--", lw=1,
                   label="10.01 deg (trajectory plane fit)")
        a2.axhline(1.47, color="tab:green", ls=":", lw=1.4,
                   label="1.47 deg (Ouster IMU at rest)")
        a2.set_xlabel("t [s]  (PX4 clock, from first message)")
        a2.set_ylabel("tilt [deg]")
        a2.legend(loc="upper right")
        a2.grid(alpha=0.3)
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())