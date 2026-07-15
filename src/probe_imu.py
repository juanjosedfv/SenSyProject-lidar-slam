"""Decode aspn_msgs/msg/MeasurementIMU straight from the rosbag, without the package.

The ASPN ICD (https://github.com/Open-PNT/ASPN-ICD) is a specification only: it
ships no .msg files, so ``aspn_msgs`` cannot be built from it and rosbag2 skips
/ouster/imu_meas entirely. The wire layout was recovered by hand from the CDR
payload: the generated message prepends a std_msgs/Header (frame_id
"ouster_imu") to the ICD fields, which is why a naive ICD-order parse is
misaligned.

Recovered layout (CDR, little-endian, alignment = field size):

     0  int32   header.stamp.sec
     4  uint32  header.stamp.nanosec
     8  uint32  len(frame_id) + 1
    12  char[]  frame_id, null-terminated ("ouster_imu")
    ..  uint32  vendor_id            (aligned to 4)
    ..  uint64  device_id            (aligned to 8)
    ..  uint32  context_id
    ..  uint16  sequence_id
    ..  int64   time_of_validity     (aligned to 8, nanoseconds)
    ..  uint8   IMU_type             (0 = INTEGRATED, 1 = SAMPLED)
    ..  float64[3] meas_accel        (aligned to 8)
    ..  float64[3] meas_gyro
    ..  uint8   num_integrity

Three independent checks confirm the decode rather than assuming it:
  * |meas_accel| must equal g (9.81 m/s^2) for a slow-moving vehicle
  * |meas_gyro| must be a plausible turn rate, not a huge number
  * meas_gyro z must track the KISS-ICP yaw rate turn for turn

That last one also reveals whether the IMU and LiDAR frames share an axis,
which decides whether the IMU gravity vector can be compared against the
trajectory-plane tilt at all.

    python3 -m src.probe_imu --bag /home/jinghao/ros2_ws/data/rosbag
    python3 -m src.probe_imu --bag ... --kiss-bag ~/ros2_ws/data/kiss_output
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path

import numpy as np

GRAVITY_MS2 = 9.813  # Berlin


def aligned(offset: int, size: int) -> int:
    """CDR aligns each primitive to its own size, relative to the stream start."""

    remainder = offset % size
    return offset if remainder == 0 else offset + (size - remainder)


class CdrReader:
    """Minimal CDR reader that tracks alignment for us."""

    def __init__(self, payload: bytes):
        self.little_endian = payload[1] in (1, 3)
        self.endian = "<" if self.little_endian else ">"
        self.body = payload[4:]  # skip the 4-byte encapsulation header
        self.offset = 0

    def _read(self, fmt: str, size: int, count: int = 1):
        self.offset = aligned(self.offset, size)
        values = struct.unpack_from(f"{self.endian}{count}{fmt}", self.body, self.offset)
        self.offset += size * count
        return values

    def uint8(self) -> int:
        return self._read("B", 1)[0]

    def uint16(self) -> int:
        return self._read("H", 2)[0]

    def int32(self) -> int:
        return self._read("i", 4)[0]

    def uint32(self) -> int:
        return self._read("I", 4)[0]

    def int64(self) -> int:
        return self._read("q", 8)[0]

    def uint64(self) -> int:
        return self._read("Q", 8)[0]

    def float64_array(self, count: int) -> np.ndarray:
        return np.array(self._read("d", 8, count), dtype="float64")

    def string(self) -> str:
        length = self.uint32()
        raw = self.body[self.offset : self.offset + length]
        self.offset += length
        return raw.rstrip(b"\x00").decode("ascii", "replace")


def decode_imu(payload: bytes) -> dict:
    """Decode one MeasurementIMU CDR payload."""

    reader = CdrReader(payload)
    stamp_sec = reader.int32()
    stamp_nsec = reader.uint32()
    frame_id = reader.string()

    vendor_id = reader.uint32()
    device_id = reader.uint64()
    context_id = reader.uint32()
    sequence_id = reader.uint16()
    time_of_validity = reader.int64()
    imu_type = reader.uint8()
    accel = reader.float64_array(3)
    gyro = reader.float64_array(3)

    return {
        "stamp_s": stamp_sec + stamp_nsec * 1e-9,
        "frame_id": frame_id,
        "vendor_id": vendor_id,
        "device_id": device_id,
        "context_id": context_id,
        "sequence_id": sequence_id,
        "time_of_validity_s": time_of_validity * 1e-9,
        "imu_type": imu_type,
        "accel": accel,
        "gyro": gyro,
    }


def read_imu(db_path: Path, topic: str, limit: int | None = None) -> list[dict]:
    query = """
        select messages.data from messages
        join topics on messages.topic_id = topics.id
        where topics.name = ?
        order by messages.timestamp
    """
    params: tuple = (topic,)
    if limit:
        query += " limit ?"
        params = (topic, limit)

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        raise ValueError(f"No messages on {topic}")
    return [decode_imu(bytes(r[0])) for r in rows]


def yaw_rate_from_odometry(kiss_bag: Path, stride: int = 5):
    """KISS-ICP yaw rate about the odom vertical, for cross-checking the gyro."""

    from scipy.spatial.transform import Rotation

    from .kiss_adapter import load_kiss_odometry

    trajectory = load_kiss_odometry(kiss_bag, verbose=False)
    times = np.array([s.timestamp_s for s in trajectory], dtype="float64")
    rotations = [np.asarray(s.transform, dtype="float64")[:3, :3] for s in trajectory]

    stamps, rates = [], []
    for i in range(0, len(rotations) - stride, stride):
        dt = times[i + stride] - times[i]
        if dt <= 0:
            continue
        # incremental rotation expressed in the *sensor* frame, so that its z
        # component is comparable with a body-mounted gyro
        delta = rotations[i].T @ rotations[i + stride]
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        stamps.append(0.5 * (times[i] + times[i + stride]))
        rates.append(rotvec / dt)
    return np.array(stamps), np.array(rates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode ASPN IMU records from a rosbag")
    parser.add_argument("--bag", required=True, help="Original rosbag directory")
    parser.add_argument("--topic", default="/ouster/imu_meas", help="IMU topic")
    parser.add_argument("--kiss-bag", default=None, help="Cross-check gyro vs KISS-ICP yaw")
    parser.add_argument("--limit", type=int, default=None, help="Decode only N messages")
    parser.add_argument("--rest-window", type=float, default=5.0,
                        help="Seconds at the start assumed stationary")
    parser.add_argument("--hex", action="store_true", help="Dump the first payload")
    args = parser.parse_args()

    bag_dir = Path(args.bag).expanduser()
    candidates = sorted(bag_dir.glob("*.db3")) if bag_dir.is_dir() else [bag_dir]
    if not candidates:
        raise FileNotFoundError(f"No .db3 in {bag_dir}")
    db_path = candidates[0]

    if args.hex:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "select messages.data from messages join topics on "
                "messages.topic_id = topics.id where topics.name = ? "
                "order by messages.timestamp limit 1",
                (args.topic,),
            ).fetchone()
        raw = bytes(row[0])
        print(f"first payload, {len(raw)} bytes:")
        for i in range(0, min(len(raw), 144), 16):
            print(f"  {i:04d}  {raw[i:i + 16].hex(' ')}")
        print()

    messages = read_imu(db_path, args.topic, args.limit)
    first = messages[0]
    stamps = np.array([m["stamp_s"] for m in messages])
    accel = np.array([m["accel"] for m in messages])
    gyro = np.array([m["gyro"] for m in messages])
    accel_norm = np.linalg.norm(accel, axis=1)

    print(f"decoded {len(messages)} messages from {args.topic}")
    print(f"  frame_id    : {first['frame_id']!r}")
    print(f"  vendor/device/context: {first['vendor_id']} / "
          f"{first['device_id']} / {first['context_id']}")
    print(f"  IMU_type    : {first['imu_type']} "
          f"({'INTEGRATED' if first['imu_type'] == 0 else 'SAMPLED'})")
    print(f"  stamps      : {stamps[0]:.3f} .. {stamps[-1]:.3f} "
          f"({len(messages) / (stamps[-1] - stamps[0]):.1f} Hz)")

    print("\n[check 1] specific force magnitude")
    print(f"  |accel| = {accel_norm.mean():.4f} +/- {accel_norm.std():.4f} m/s^2 "
          f"(expect ~{GRAVITY_MS2})")
    ok = abs(accel_norm.mean() - GRAVITY_MS2) < 0.5
    print(f"  -> {'DECODE CONFIRMED' if ok else 'still misaligned'}")
    if not ok:
        print("     re-run with --hex and check the byte offsets")
        return

    print("\n[check 2] angular rate magnitude")
    gyro_norm = np.linalg.norm(gyro, axis=1)
    print(f"  |gyro|  = {np.degrees(gyro_norm.mean()):.2f} +/- "
          f"{np.degrees(gyro_norm.std()):.2f} deg/s (plausible for a slow vehicle)")

    rest = stamps <= stamps[0] + args.rest_window
    print(f"\n[gravity] averaged over the first {args.rest_window:g} s "
          f"({int(rest.sum())} samples)")
    mean_accel = accel[rest].mean(axis=0)
    gravity = mean_accel / np.linalg.norm(mean_accel)
    tilt = np.degrees(np.arccos(np.clip(abs(gravity[2]), 0.0, 1.0)))
    print(f"  specific force : [{mean_accel[0]:+.4f} {mean_accel[1]:+.4f} "
          f"{mean_accel[2]:+.4f}] m/s^2")
    print(f"  unit direction : [{gravity[0]:+.5f} {gravity[1]:+.5f} {gravity[2]:+.5f}]")
    print(f"  tilt from IMU z-axis : {tilt:.2f} deg")
    print(f"  rest check: |accel| std over the window = "
          f"{accel_norm[rest].std():.4f} m/s^2, "
          f"|gyro| = {np.degrees(gyro_norm[rest].mean()):.2f} deg/s")

    if args.kiss_bag is None:
        print("\nPass --kiss-bag to cross-check the gyro against the KISS-ICP yaw rate;")
        print("that is what decides whether the IMU and LiDAR axes are aligned, and")
        print("therefore whether this tilt is comparable with the trajectory plane fit.")
        return

    print("\n[check 3] gyro vs KISS-ICP angular rate (same axis?)")
    odom_t, odom_rates = yaw_rate_from_odometry(Path(args.kiss_bag).expanduser())
    inside = (odom_t >= stamps[0]) & (odom_t <= stamps[-1])
    if inside.sum() < 10:
        print("  no time overlap with the odometry bag")
        return

    odom_t, odom_rates = odom_t[inside], odom_rates[inside]
    resampled = np.column_stack(
        [np.interp(odom_t, stamps, gyro[:, axis]) for axis in range(3)]
    )
    for axis, name in enumerate("xyz"):
        a, b = resampled[:, axis], odom_rates[:, axis]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        correlation = float(np.corrcoef(a, b)[0, 1])
        scale = float(np.polyfit(b, a, 1)[0])
        print(f"  {name}: correlation {correlation:+.3f}, gyro/odom scale {scale:+.3f}")
    print("  -> a correlation near +/-1 on z means the two frames share the vertical,")
    print("     so the IMU gravity vector and the plane-fit tilt describe the same thing")


if __name__ == "__main__":
    main()