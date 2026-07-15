# KISS-ICP LiDAR Odometry Pipeline

LiDAR-based positioning for the SenSy26 XTrack dataset, using
[KISS-ICP](https://github.com/PRBonn/kiss-icp) as the odometry engine and the
project's existing evaluation stack (`geo.py`, `evaluation.py`) unchanged.

This complements the LAZ + hand-written ICP pipeline in `main.py`: both share the
same coordinate maths, GNSS handling, metrics and plots, and differ only in where
the trajectory comes from. Having the two side by side is deliberate — the
hand-written ICP is the baseline the KISS-ICP result is judged against.

## Results

| Metric | Raw | Gravity-levelled (`--level`) |
| --- | --- | --- |
| Horizontal RMSE vs corrected GNSS | 2.72 m | **2.26 m** |
| Median error | 2.33 m | **1.64 m** |
| 95th percentile | 4.53 m | 4.00 m |
| Max error | 8.50 m | 8.43 m |
| Path length ratio (LiDAR / GNSS) | 0.981 | **0.993** |
| Relative drift | 0.47 % | **0.39 %** |

7771 poses over 819.5 s, 6454 matched against GNSS. For reference, KISS-ICP
reports ~0.5 % translational error on KITTI, and the GNSS ground truth carries
0.86 m mean horizontal uncertainty (`eph`) of its own.

Deliverables land in `outputs/`:

| File | Content |
| --- | --- |
| `trajectory_estimate.csv` | per-pose timestamp, local XYZ, ENU, **lat/lon/alt** |
| `velocity_ned.csv` | timestamp, **vn / ve / vd** [m/s] |
| `map.pcd` | global point-cloud map (475 843 points at 0.25 m voxels, 14 MB) |
| `metrics.json` | RMSE, percentiles, path lengths, tilt, clock offset |
| `plots/trajectory_vs_gnss.png` | estimated trajectory over GNSS ground truth |
| `plots/horizontal_error.png` | horizontal error against time |
| `plots/reference_diagnostics.png` | raw odometry frame vs GNSS frame |
| `plots/vertical_check.png` | odometry z vs GNSS altitude |

## Environment

Ubuntu 22.04 + ROS 2 Humble, in WSL2. Everything below assumes both overlays are
sourced; putting them in `~/.bashrc` saves repeating them:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### 1. ROS 2 workspace

```bash
mkdir -p ~/ros2_ws/src ~/ros2_ws/data
```

### 2. px4_msgs (required — the bag is full of PX4 types)

Without it `ros2 bag info` cannot even name the `/fmu/out/*` topics.

```bash
cd ~/ros2_ws/src
git clone -b release/1.17 https://github.com/PX4/px4_msgs.git
cd ~/ros2_ws && colcon build --packages-select px4_msgs
```

### 3. CMake (Ubuntu 22.04 ships 3.22.1, which is too old for KISS-ICP v1.3.0)

```bash
pip install --upgrade cmake && hash -r
cmake --version          # expect >= 3.28
```

### 4. KISS-ICP

```bash
cd ~/ros2_ws/src && git clone https://github.com/PRBonn/kiss-icp
cd ~/ros2_ws && colcon build --packages-select kiss_icp
ros2 pkg list | grep kiss_icp    # verify
```

Warnings during the build are normal; errors are not.

### 5. Python

`numpy`, `scipy`, `pyyaml` and `matplotlib` already come from the ROS 2 apt
packages — **do not `pip install` them.** See "Gotchas".

### 6. Data

The bag is 23.5 GB, so symlink rather than copy:

```bash
ln -s /mnt/c/SensorSystems/rosbag ~/ros2_ws/data/rosbag
ros2 bag info ~/ros2_ws/data/rosbag
```

Relevant topics:

| Topic | Type | Use |
| --- | --- | --- |
| `/ouster/points` | `sensor_msgs/PointCloud2` | KISS-ICP input (8194 scans, ~10 Hz) |
| `/fmu/out/vehicle_gps_position` | `px4_msgs/SensorGps` | raw GNSS |
| `/ouster/imu_meas` | `aspn_msgs/MeasurementIMU` | IMU (see Diagnostics) |

Ground truth is the corrected GNSS CSV, `xtrack_global_position_t12.csv`
(7686 fixes, `eph` 0.86 m, `epv` 0.39 m).

## Running

### Step 1 — KISS-ICP, recording the odometry as you go

`/kiss/odometry` only exists while the node runs, so the recorder has to be
running at the same time. Two terminals:

```bash
# terminal 1 — note the ABSOLUTE path, see Gotchas
ros2 launch kiss_icp odometry.launch.py \
    bagfile:=/home/jinghao/ros2_ws/data/rosbag \
    topic:=/ouster/points
```

```bash
# terminal 2 — start once RViz is up and points are moving
cd ~/ros2_ws/data
ros2 bag record /kiss/odometry -o kiss_output
```

Takes ~30 min (I/O-bound on `/mnt/c`, not CPU-bound). Stop terminal 2 with
Ctrl+C *after* terminal 1 finishes. The result is ~6 MB:

```bash
ros2 bag info ~/ros2_ws/data/kiss_output   # expect ~7771 messages
```

Only `/kiss/odometry` is recorded — the point-cloud topics would add tens of GB,
and the map is rebuilt from the poses in step 3 anyway.

### Step 2 — evaluate against GNSS

```bash
cd ~/SenSyProject
python3 -m src.pipeline_kiss \
    --kiss-bag /home/jinghao/ros2_ws/data/kiss_output \
    --gnss-csv /mnt/c/SensorSystems/xtrack_gnss_corrected/xtrack_global_position_t12.csv \
    --level
```

Writes the trajectory CSV, velocity CSV, metrics and plots.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--level` | gravity-align the odometry frame (see Findings) |
| `--scan-offset` | brute-force the LiDAR/GNSS clock offset instead of trusting the config |
| `--gnss-time-offset` | override the offset explicitly |

### Step 3 — build the map

```bash
python3 -m src.build_map \
    --bag /home/jinghao/ros2_ws/data/rosbag \
    --kiss-bag /home/jinghao/ros2_ws/data/kiss_output \
    --level --max-range 25
```

Streams `/ouster/points` from the bag, looks each scan's pose up by header stamp,
transforms it and accumulates into a voxel grid. Reusing the recorded poses means
the map is consistent with the trajectory CSV by construction, and KISS-ICP does
not have to be re-run.

`--stride 10` (default) uses every 10th scan; `--voxel 0.25` sets the resolution;
`--max-range` gates out the noisy far returns. Memory stays bounded because the
grid only ever stores occupied voxels, not the 16 M raw points.

View the result in CloudCompare (`\\wsl.localhost\Ubuntu-22.04\home\jinghao\SenSyProject\outputs`),
then Edit → Colors → Height Ramp.

## Module map

| File | Role |
| --- | --- |
| `kiss_adapter.py` | reads `/kiss/odometry` from the recorded bag into `OdometryStep`s — the only translation layer between KISS-ICP and the existing evaluation code |
| `pipeline_kiss.py` | orchestrates estimate → GNSS alignment → RMSE → CSV/plots |
| `build_map.py` | accumulates the global `.pcd` from the recorded poses |
| `level.py` | gravity alignment; also a standalone tilt diagnostic |
| `probe_imu.py` | decodes `aspn_msgs/MeasurementIMU` straight from the bag |
| `check_vertical.py` | checks the odometry vertical against GNSS altitude |
| `evaluation.py`, `geo.py` | **unchanged**, shared with `main.py` |
| `odometry.py`, `preprocess.py` | `OdometryStep`, `write_ascii_pcd`, `filter_points` reused; the hand-written ICP is the baseline |

## Findings

### The clock offset is real and independently confirmed

The LiDAR and PX4 clocks are not synchronised; `config.yaml` carries
`gnss_time_offset_s: -312.7144`, calibrated on the LAZ pipeline. `--scan-offset`
re-derives it from the bag header stamps alone and lands on the same value:

```
offset [s]   matched   RMSE [m]
   -322.71      6364       8.15
   -317.71      6411       4.07
   -312.71      6454       2.72   <-- minimum
   -307.71      6502       6.29
```

A clean V with the minimum exactly on the calibrated value confirms the LAZ
timestamps and the bag's `header.stamp` share one clock.

### The odometry frame is tilted ~10 deg, and it is *not* the sensor mount

KISS-ICP has no IMU, so `odom_lidar` inherits the sensor attitude of the first
scan and has no notion of gravity. The trajectory turns out to lie on a plane
tilted 10 deg from the odom XY plane — driving on flat ground shows up as a 42 m
climb.

Three measurements, three different data sources:

| Method | Data used | Tilt |
| --- | --- | --- |
| Plane fit through trajectory positions | positions | 10.01 deg (0.97 m RMS residual) |
| Dominant rotation axis of the orientations | quaternions | 10.82 deg |
| Ouster IMU gravity vector at rest | accelerometer | **1.47 deg** |

The first two agree, but the IMU — the only one that measures gravity directly
rather than assuming flat ground — says the sensor is essentially upright. So the
tilt is *not* a mounting angle: the odometry itself is systematically wrong in the
vertical.

GNSS altitude settles it (`check_vertical.py`): `epv` is 0.39 m with 0.6 cm
sample-to-sample jitter, so the altitude channel is trustworthy, and it says the
terrain really does vary by 12.9 m — about 1 % average grade over the 1290 m
route, which is imperceptible on foot. Regressing `alt ~ a*x + b*y + c*z + d`
over the odometry positions gives |(a,b,c)| = 0.26 and R² = 0.22: the odometry
positions barely predict real altitude at all, and its z correlates with GNSS
altitude at only +0.14.

Compensating the tilt still improves the **horizontal** result (RMSE 2.72 → 2.26 m,
median 2.33 → 1.64 m) because the 2D rigid alignment cannot absorb the
anisotropic cos(10°) scale distortion the tilt introduces. Note the gradient:
the median improves 30 % while the maximum barely moves — systematic distortion
is removed, the error spikes (local ICP failures at turns) are not.

The vertical channel of `map.pcd` and `trajectory_estimate.csv` should therefore
be treated as unreliable. Fixing it needs a gravity reference, i.e. IMU fusion —
which is precisely the "Inertial" half of LiDAR-Inertial odometry.

### Scan deskewing is disabled

```
Field 't', 'timestamp', 'time_stamp', or 'time' does not exist. Disabling scan deskewing
```

The recorded clouds carry no per-point timestamps, so KISS-ICP cannot compensate
the motion within a scan. At 1.57 m/s and 3.63 deg/s (IMU-measured), that is
15.7 cm and 0.36 deg of intra-scan distortion — real, but two orders of magnitude
too small to explain the 10 deg tilt above.

### The IMU is readable without `aspn_msgs`

`/ouster/imu_meas` uses `aspn_msgs/MeasurementIMU`, which cannot be built:
[ASPN-ICD](https://github.com/Open-PNT/ASPN-ICD) is a *specification* and ships
no `.msg` files, so rosbag2 skips the topic entirely. `probe_imu.py` decodes the
CDR payload directly. The generated message prepends a `std_msgs/Header`
(`frame_id: "ouster_imu"`) to the ICD fields, which is why a naive ICD-order
parse misaligns.

The decode is validated, not assumed: |accel| = 9.72 m/s² against Berlin's 9.813,
|gyro| = 3.63 deg/s, and the gyro correlates with the KISS-ICP yaw rate at -0.874
(the sign pattern shows the IMU and LiDAR frames differ by 180 deg about X).

## Gotchas

Things that cost real time, recorded so they cost it only once.

- **`~` is not expanded in launch arguments.** `bagfile:=~/ros2_ws/data/rosbag`
  fails with `Bag path '~/ros2_ws/data/rosbag' does not exist!` — the launch file
  passes the string through verbatim. Use the absolute path.
- **Never `pip install` into the ROS environment.** ROS 2's Python packages come
  from apt and are compiled against NumPy 1.x. `pip install open3d` pulls NumPy
  2.x into `~/.local`, which shadows the system one and breaks SciPy's ABI —
  taking `main.py`'s `cKDTree` down with it. Recovery:
  `rm -rf ~/.local/lib/python3.10/site-packages`. Use a venv, or apt packages.
- **Use `bagfile:=`, not a separate `ros2 bag play`.** Playing a 23.5 GB bag from
  `/mnt/c` in real time starves the queue (`Message queue starved. Messages will
  be delayed.`) and drops point clouds. With `bagfile:=` playback is paced by the
  node, so nothing is lost — it just takes longer.
- **Use `header.stamp`, never the bag receive time.** The recorded odometry's bag
  timestamps are the wall clock of *when you recorded* (43 days after the data was
  collected). Only `header.stamp`, inherited from `/ouster/points`, can be matched
  against GNSS. `kiss_adapter.py` reports the gap so the mistake is visible.
- **`aspn_msgs not found` on `bag play` is harmless.** The two IMU topics are
  skipped; `/ouster/points` is a standard type and plays fine.
- **`resolve_dataset_paths` requires `laz_clouds/` to exist.** That is why
  `pipeline_kiss.py` reads its paths directly instead of reusing it — the
  KISS-ICP route consumes the bag and never needs the LAZ export.

## Diagnostics

```bash
# frame tilt, both estimators, swept over stride to separate noise from bias
python3 -m src.level --kiss-bag ~/ros2_ws/data/kiss_output

# decode the IMU and cross-check it against the odometry
python3 -m src.probe_imu --bag ~/ros2_ws/data/rosbag \
    --kiss-bag ~/ros2_ws/data/kiss_output --limit 20000

# odometry vertical vs GNSS altitude
python3 -m src.check_vertical \
    --kiss-bag ~/ros2_ws/data/kiss_output \
    --gnss-csv /mnt/c/SensorSystems/xtrack_gnss_corrected/xtrack_global_position_t12.csv
```

## Known limitations

- **Vertical is unreliable** — see Findings. Horizontal results are unaffected.
- **RMSE is measured after 2D rigid alignment**, so it scores trajectory *shape*,
  not absolute geo-referencing. This is standard for odometry (the starting
  heading is arbitrary), but it does mean the number is not a positioning
  accuracy in the GNSS sense.
- **83 % coverage (6454 / 7771 poses).** After the clock offset, the first ~145 s
  of the trajectory falls outside the GNSS record; those poses cannot be scored.
- **Deskewing is off**, costing some accuracy at speed.
- **Gravity alignment assumes level ground.** The plane-fit residual (0.97 m RMS)
  supports it, but GNSS altitude shows it is an approximation.