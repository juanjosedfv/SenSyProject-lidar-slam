# KISS-ICP LiDAR Odometry Pipeline

LiDAR-based positioning for the SenSy26 XTrack dataset, using
[KISS-ICP](https://github.com/PRBonn/kiss-icp) as the odometry engine and the
project's existing evaluation stack (`geo.py`, `evaluation.py`) unchanged.

This complements the LAZ + hand-written ICP pipeline in `main.py`: both share the
same coordinate maths, GNSS handling, metrics and plots, and differ only in where
the trajectory comes from. Having the two side by side is deliberate — the
hand-written ICP is the baseline the KISS-ICP result is judged against.

**Every position here comes from LiDAR scan matching.** The bag also contains
PX4 EKF2's full state estimate (`/fmu/out/vehicle_odometry`, 81386 messages:
position, velocity, attitude, covariance at 100 Hz) — that is the answer to the
task, and it is deliberately not used. `/fmu/out/vehicle_attitude` is read by
one diagnostic (`probe_attitude.py`) to validate an assumption; it does not
enter the estimate either.

## Scope: which data this covers

**The bag is dataset 1 only. The corrected GNSS CSV is both datasets.** The
project intro says so: *"one file for both datasets, filter by timestamp"*.

| Record | Span | Content |
| --- | --- | --- |
| `rosbag` (`/ouster/points`, 8194 scans) | 819.5 s | **Dataset 1 only**: ILR → Ernst-Reuter-Platz and back |
| `xtrack_global_position_t12.csv` (7686 fixes) | 1552 s | **Both**: dataset 1, then dataset 2's loop around the ILR / math building |

Plotting the CSV shows it directly: a north–south corridor (dataset 1, the only
part any LiDAR pose matches) plus a large eastward loop the LiDAR never
recorded. `/fmu/out/vehicle_gps_position` in the bag has 8223 fixes over 822 s —
the bag's own GNSS covers dataset 1 and nothing else.

**RMSE is unaffected**: matching is by timestamp, so dataset 2's fixes never
match a LiDAR pose, and the reported GNSS path (575.2 m) is over the 6454
matched poses only. But **any statistic over the whole CSV is meaningless
here** — its 1290.4 m "path length" is two routes added together.

## Results

| Metric | Raw | Mount-compensated (`--level`) |
| --- | --- | --- |
| Horizontal RMSE vs corrected GNSS | 2.72 m | **2.26 m** |
| Median error | 2.33 m | **1.64 m** |
| 95th percentile | 4.53 m | 4.00 m |
| Max error | 8.50 m | 8.43 m |
| LiDAR path length | 564.5 m | 571.1 m |
| Path length ratio (LiDAR / GNSS, matched subset) | 0.981 | **0.993** |
| Relative drift (RMSE / GNSS path) | 0.47 % | **0.39 %** |
| Vertical closure error (return to start) | 3.23 m | 3.23 m |

7771 poses over 819.5 s, 6454 matched against GNSS over 575.2 m of GNSS path.
For reference, KISS-ICP reports ~0.5 % translational error on KITTI, and the
GNSS ground truth carries 0.86 m mean horizontal uncertainty (`eph`) of its own.

Deliverables land in `outputs/`:

| File | Content |
| --- | --- |
| `trajectory_estimate.csv` | per-pose timestamp, local XYZ, ENU, lat/lon/alt |
| `velocity_ned.csv` | timestamp, vn / ve / vd [m/s] |
| `map.pcd` | global point-cloud map (475 843 points at 0.25 m voxels, 14 MB) |
| `metrics.json` | RMSE, percentiles, path lengths, tilt, clock offset |
| `plots/trajectory_vs_gnss.png` | estimated trajectory over GNSS ground truth |
| `plots/horizontal_error.png` | horizontal error against time |
| `plots/reference_diagnostics.png` | raw odometry frame vs GNSS frame |
| `plots/vertical_check.png` | odometry z vs GNSS altitude |
| `plots/pixhawk_attitude.png` | Pixhawk EKF2 roll/pitch/tilt, 100 Hz (diagnostic) |

`lat` and `lon` are sound. **`alt` and `vd` are not**: the GNSS alignment is
2D-only and passes z through untouched, so `up_m` is bit-identical to
`z_lidar_m`, `alt` is literally a constant (24.594088) plus `up_m`, and `vd` is
its derivative. They inherit the odometry's vertical error — see Known
limitations.

## Environment

Ubuntu 22.04 + ROS 2 Humble (**Python 3.10**), in WSL2. Everything below assumes
both overlays are sourced; putting them in `~/.bashrc` saves repeating them:

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

| Topic | Type | Count | Use |
| --- | --- | --- | --- |
| `/ouster/points` | `sensor_msgs/PointCloud2` | 8194 | KISS-ICP input (~10 Hz) |
| `/fmu/out/vehicle_attitude` | `px4_msgs/VehicleAttitude` | 81799 | diagnostic only — validates the `--level` premise |
| `/fmu/out/vehicle_gps_position` | `px4_msgs/SensorGps` | 8223 | raw GNSS, on the bag's clock |
| `/ouster/imu_meas` | `aspn_msgs/MeasurementIMU` | — | raw IMU (see Diagnostics) |

Also present and deliberately unused: `/fmu/out/vehicle_odometry` (81386) and
`/fmu/out/vehicle_local_position_v1` (41105).

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
Ctrl+C *after* terminal 1 finishes.

```bash
ros2 bag info ~/ros2_ws/data/kiss_output   # 7771 messages, not 8194 — see Gotchas
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

| Flag | Effect |
| --- | --- |
| `--level` | compensate the LiDAR mount rotation (see Findings) |
| `--scan-offset` | brute-force the CSV/LiDAR clock offset instead of trusting the config |
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
`--max-range` gates out the noisy far returns (the OS0-32 is a short-range
sensor — the intro quotes `< 100 m`, and usable range is well under that).
Memory stays bounded because the grid only ever stores occupied voxels.

**`--level` matters here as much as for the trajectory**: without it the map is
built in a frame tilted 10°, so flat ground comes out as a ramp.

View the result in CloudCompare (`\\wsl.localhost\Ubuntu-22.04\home\jinghao\SenSyProject\outputs`),
then Edit → Colors → Height Ramp.

## Module map

| File | Role |
| --- | --- |
| `kiss_adapter.py` | reads `/kiss/odometry` from the recorded bag into `OdometryStep`s — the only translation layer between KISS-ICP and the existing evaluation code |
| `pipeline_kiss.py` | orchestrates estimate → GNSS alignment → RMSE → CSV/plots |
| `build_map.py` | accumulates the global `.pcd` from the recorded poses |
| `level.py` | mount-rotation compensation, and the standalone tilt diagnostic (two independent estimators) |
| `probe_attitude.py` | Pixhawk EKF2 attitude — validates `level.py`'s premise |
| `probe_imu.py` | decodes `aspn_msgs/MeasurementIMU` straight from the bag |
| `check_vertical.py` | checks the odometry vertical against GNSS altitude |
| `evaluation.py`, `geo.py` | **unchanged**, shared with `main.py` |
| `odometry.py`, `preprocess.py` | `OdometryStep`, `write_ascii_pcd`, `filter_points` reused; the hand-written ICP is the baseline |

## Findings

### The LiDAR is mounted ~10° off vertical, and that is the whole story

KISS-ICP has no absolute attitude reference, so `odom_lidar` is simply the
sensor frame of the very first scan. The Ouster on the XTrack is mounted at an
angle — pitching a LiDAR down to see more ground is standard on a ground
vehicle — so that frame is tilted with respect to gravity and the trajectory and
map come out tilted with it.

**Two independent estimators, two different data sources, one number:**

| Method | Data used | Assumes | Tilt |
| --- | --- | --- | --- |
| Plane fit through the trajectory | **positions** | ground is level | **10.01°** |
| Dominant rotation axis of the orientations | **quaternions** | vehicle turns about gravity | **9.98°** |

**0.03° apart.** Positions and orientations are different quantities; there is
no mechanism by which an *error* would make them agree to a hundredth of a
degree. They agree because they are measuring the same physical constant.

The rotation-axis estimator is swept over stride to separate noise from bias
(`convergence means noise, drift means bias`):

```
 stride  turns  tot deg    tilt   conc  scatter  dir SE  vs plane
     10    437     2507  10.82d  0.888    24.6d   1.18d     5.35d
     25    183     1673  11.44d  0.930    22.3d   1.65d     6.36d
     50     95     1187   9.98d  0.954    20.2d   2.08d     5.44d
    100     48      635   9.97d  0.938    24.3d   3.51d     6.12d
    200     26      436  10.32d  0.958    17.6d   3.45d     6.09d

tilt across strides: 10.51 +/- 0.56 deg  -> converged, no drift
```

**It is a frame rotation, not drift.** The plane-fit residual is 0.97 m RMS
against a 42.43 m z span — **2.3 %**. Vertical *drift* would accumulate with
time, so the same place visited outbound (t≈250 s) and inbound (t≈670 s) would
differ by tens of metres and no plane would fit that tightly. It fits, so z is a
function of *position*, which is the signature of a tilted frame.

**The "42 m climb" is arithmetic, not a climb:**

```
measured raw z span                      42.43 m
240.7 m corridor  x  tan(10.01 deg)  =   42.49 m     (0.1 % apart)
```

`--level` rotates it away entirely: z then spans only [−3.25, +0.18] m.

### The premise is confirmed by a third sensor

The rotation-axis estimator assumes only that the platform turns about gravity.
`/fmu/out/vehicle_attitude` (PX4 EKF2, accel + gyro + mag fused, 81799 samples
over 822.3 s) measures that directly:

```
           median      mean        p5       p95       min       max
roll         0.46      0.09     -2.55      2.00     -4.89      4.03
pitch       -0.33     -0.18     -1.57      1.76     -5.84      5.27
TILT         1.73      1.78      0.46      2.81      0.01      5.88

> 10 deg :   0.0 %   of the record
```

The platform is level throughout — it never reaches 10° at any instant. **So its
turns are about gravity, and the 9.98° really is the angle between the LiDAR's
z axis and vertical.** This is what discharges the assumption behind `--level`:
it is a measured extrinsic compensation, not a guess.

### Correction: the Ouster IMU's 1.47° is not evidence, the decode is wrong

`probe_imu.py` reports the Ouster's internal IMU as 1.47° from vertical, which
was previously read as "the sensor is upright, so the 10° must be odometry
error". That reading does not survive scrutiny:

- **The Ouster IMU is bolted inside the Ouster housing.** If the LiDAR is
  mounted 10° off vertical, its IMU is too. Reading 1.47° means the decode is
  wrong, not that the LiDAR is level.
- **Ouster's frames**: lidar → sensor is a **180° rotation about Z** (plus a Z
  translation); the IMU shares the sensor frame's orientation. 180° about Z
  leaves the z axis alone, so gyro-z and the LiDAR yaw rate should share a sign.
  The measured correlation is **−0.874**. Something in the hand-rolled CDR
  decode has an axis or sign problem.
- **The previous claim that the frames differ by 180° about X was inferred from
  that −0.874**, i.e. from the suspect decode itself. It contradicts the
  documented 180° about Z.
- **`|accel| = 9.72` against Berlin's 9.813 does not validate the axes.**
  Magnitude is invariant under any rotation, so the check confirms that three
  plausible float32s were found and nothing more.

`probe_imu.py` is kept as a decoding exercise. **Its tilt figure should not be
quoted.** The Pixhawk's fused solution is the reliable inertial reference.

### The clock offset is real, but it is not a hardware clock difference

`config.yaml` carries `gnss_time_offset_s: -312.7144`. `--scan-offset` re-derives
it from the bag header stamps alone and lands on the same value:

```
offset [s]   matched   RMSE [m]
   -322.71      6364       8.15
   -317.71      6411       4.07
   -312.71      6454       2.72   <-- minimum
   -307.71      6502       6.29
```

A clean V with the minimum exactly on the calibrated value. **The number is
correct.** The explanation previously given here — that the LiDAR and PX4 clocks
are not synchronised — is **false**:

| Source | First stamp | Clock |
| --- | --- | --- |
| `/fmu/out/vehicle_attitude` | 1780397390.96 | PX4, **in the bag** |
| `/ouster/points` `header.stamp` | 1780397393.71 | Ouster, **in the bag** |
| corrected GNSS CSV | 1780397225.80 | reconstructed from the Pixhawk log |

**PX4 and Ouster differ by 2.75 s inside the bag** — that is when each topic
started streaming. If the clocks really differed by 313 s, so would these.

So the 312.7144 s is a property of **the corrected CSV's reconstructed time base
versus the bag's**, not of the sensors. The CSV comes from the `.ulg` (PX4
`hrt_absolute_time()` plus a reconstructed epoch); the bag's `/fmu/out/*` stamps
were mapped onto the host clock by the uXRCE-DDS timesync.

Ruled out: reading the wrong column. The intro notes *"the timestamp_sample is
the same as you have in the rosbags"*, which suggests `timestamp` is not — but
measured, `timestamp_sample - timestamp = -0.00036 s`. Not it. (The pipeline
already uses `timestamp_sample`.)

### 423 scans (5.2 %) are dropped

`/ouster/points` has 8194 scans; `/kiss/odometry` has 7771 poses — over the
**same 819.5 s span**, i.e. 9.48 Hz against 10 Hz. Same span, lower rate, so the
losses are distributed, not a late-started recorder. Visible directly in
`trajectory_estimate.csv` as 0.2–0.3 s gaps where 0.1 s is expected.

Cause: `OdometryServer.cpp:87` subscribes with `rclcpp::SensorDataQoS()` —
**BEST_EFFORT, KEEP_LAST(5)**. Real-time playback of dense clouds from `/mnt/c`
outruns the node, and the QoS permits silent discard. **The loss is
non-deterministic**, so repeated runs of the same config differ — an unmeasured
noise floor under any parameter comparison.

### 41 % of the record is stationary

The intro describes dataset 1 as *"straight movements, **stops**, and turns"*.
`plots/pixhawk_attitude.png` shows it plainly: smooth roll/pitch traces alternate
with ±3–4° vibration, and **vibration is the motion signal**.

| t [s] | Attitude | State |
| --- | --- | --- |
| 0 → ~148 | smooth | stationary |
| 148 → ~330 | ±3–4° vibration | driving |
| ~330 → ~425 | smooth | stopped |
| 425 → ~590 | vibration | driving |
| ~590 → ~610 | smooth | stopped |
| 610 → ~750 | vibration | driving |
| ~750 → 822 | smooth | stopped |

Roughly **335 s of 822 s**. Note this detector never touches the odometry, so it
is not self-referential — a detector built on odometry speed measures ICP jitter
and reports motion where there is none.

**Speed figures previously quoted here were wrong.** "1.57 m/s" was
1290 m ÷ 819.5 s: the *two-dataset* GNSS path over the *one-dataset* LiDAR
duration. The LiDAR segment is 564.5 m over 819.5 s = **0.69 m/s** including
stops; moving-only is roughly 1.1–1.2 m/s.

### Scan deskewing is disabled, and the config switch cannot enable it

```
Field 't', 'timestamp', 'time_stamp', or 'time' does not exist. Disabling scan deskewing
```

The chain: `Utils.hpp:93 GetTimestampField()` scans `PointCloud2.fields` for
`t`/`timestamp`/`time`/`time_stamp`, warns **once**, returns empty →
`Utils.hpp:189 GetTimestamps()` returns `{}` → `Preprocessing.cpp:59`:

```cpp
if (!deskew_ || timestamps.empty()) { return frame; }
```

**`data.deskew: True` is irrelevant.** Two independent conditions; either one
skips motion compensation, silently after the first warning. The recorded clouds
carry no per-point time field.

Magnitude: at ~1.1 m/s and 3.63 deg/s, intra-scan distortion is roughly **11 cm
and 0.36 deg** — real, but two orders of magnitude too small to matter against
the 10° mount angle. *(The previously quoted 15.7 cm came from the bogus
1.57 m/s.)*

## Gotchas

Things that cost real time, recorded so they cost it only once.

- **`~` is not expanded in launch arguments.** `bagfile:=~/ros2_ws/data/rosbag`
  fails with `Bag path '~/ros2_ws/data/rosbag' does not exist!` — the launch file
  passes the string through verbatim. Use the absolute path.
- **`bagfile:=` does not throttle playback, and does not prevent drops.** The
  launch file simply runs
  `ExecuteProcess(["ros2","bag","play","--rate","1", bag, "--clock","1000.0"])`.
  There is no back-pressure from the node; 423 of 8194 scans were lost anyway.
  The only real difference from a manual `ros2 bag play` is `--clock 1000.0`
  plus the launch's `use_sim_time:=true` default: launch without `bagfile:=` and
  then play manually **without `--clock`** and the node runs on sim time with
  nobody publishing `/clock`, so `tf2_ros::Buffer` (constructed from
  `this->get_clock()`) freezes at zero and RViz/TF break. The odometry itself
  still computes correctly — `odom_msg.header.stamp` comes from the cloud's
  header, and with `base_frame` empty `LookupTransform` is never called — but it
  looks broken. Add `--clock`, or set `use_sim_time:=false`.
- **`Message queue starved` is a player warning, not frame loss.** It means the
  reader cannot sustain real-time playback off `/mnt/c` and messages will be
  *delayed*. Drops come from the subscriber's QoS. To reduce them, lower
  `--rate`.
- **Never `pip install` into the ROS environment.** ROS 2's Python packages come
  from apt and are compiled against NumPy 1.x. `pip install open3d` (or
  `kiss-icp`, which pulls NumPy 2.5) into `~/.local` shadows the system NumPy
  and breaks SciPy's ABI — taking `main.py`'s `cKDTree` down with it. Recovery:
  `rm -rf ~/.local/lib/python3.10/site-packages`. Use a **clean venv**
  (`python3 -m venv ~/venv`, invoked as `~/venv/bin/python`), or apt packages.
  **Do not pass `--system-site-packages`**: the venv's NumPy 2.x would shadow
  the apt NumPy 1.x while the apt SciPy stays visible, reproducing the exact ABI
  break inside the venv.
- **ROS Humble is Python 3.10.** Code that compiles on 3.12 can fail here —
  PEP 701 f-strings (nested same-type quotes, multi-line expressions inside
  `{}`) are 3.12+. `ast.parse(feature_version=(3,10))` does **not** catch this;
  only a real 3.10 interpreter does.
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
# frame tilt: two independent estimators, swept over stride
python3 -m src.level --kiss-bag ~/ros2_ws/data/kiss_output

# Pixhawk EKF2 attitude: confirms the platform turns about gravity, which is
# what makes the rotation-axis estimator above trustworthy. Needs no clock offset.
python3 -m src.probe_attitude --bag ~/ros2_ws/data/rosbag

# decode the ASPN IMU (see the correction in Findings before quoting its tilt)
python3 -m src.probe_imu --bag ~/ros2_ws/data/rosbag \
    --kiss-bag ~/ros2_ws/data/kiss_output --limit 20000

# odometry vertical vs GNSS altitude
python3 -m src.check_vertical \
    --kiss-bag ~/ros2_ws/data/kiss_output \
    --gnss-csv /mnt/c/SensorSystems/xtrack_gnss_corrected/xtrack_global_position_t12.csv
```

## Known limitations

- **`--level` assumes the ground is level**, because it uses the plane fit
  through the positions. The rotation-axis estimator in `level.py` does not make
  that assumption and agrees to 0.03° in magnitude, which is why the mount angle
  itself is trusted. The two differ by ~32° in *azimuth*, of which roughly 5° is
  the plane fit's error: the trajectory is 45.3 × 240.7 m (5.3:1), so the
  across-track plane coefficient is the poorly conditioned one. `level.py`
  exposes `method="axes"` if that matters; it is not used by default.
- **The vertical does not track terrain** (R² = 0.22, correlation +0.14 with
  GNSS altitude), and carries 3.23 m of closure error — the trajectory returns
  to within 1.53 m in XY of its start but 3.23 m below it, and on a closed path
  the true Δz is ~0. That is 0.56 % of path, against 0.47 % horizontal, so it is
  ordinary odometry drift rather than a defect; but it is comparable to the
  terrain relief over this corridor, which is why the correlation is poor.
  `alt` and `vd` in the deliverables inherit it.
- **`check_vertical.py` reads the whole CSV**, i.e. both datasets. Its terrain
  figures may include dataset 2's route and are not quoted here.
- **RMSE is measured after 2D rigid alignment**, so it scores trajectory *shape*,
  not absolute geo-referencing. Standard for odometry (the starting heading is
  arbitrary), but the number is not a positioning accuracy in the GNSS sense.
- **83 % coverage (6454 / 7771 poses)**, and the cause is the offset, not the
  data. The LiDAR window (1780397393.7 .. 1780398213.2) sits *entirely inside*
  the GNSS window (1780397225.8 .. 1780398777.8) — with no offset every pose
  would have a match. Applying −312.7144 s pushes the first 144.8 s before the
  CSV's first fix. `/fmu/out/vehicle_gps_position` (8223 fixes, on the bag's
  clock, no offset needed) would cover all 822 s, at the cost of using the raw
  rather than the filtered GNSS solution; the intro recommends the filtered one
  and warns of *"noise / jumps at the beginning and end"*, so this is not done.
- **5.2 % of scans are dropped** and the loss is non-deterministic.
- **Deskewing is off**, costing ~11 cm of intra-scan distortion at speed.
- **`min_range: 0.0`** (the KISS-ICP default) feeds returns off the platform
  itself to ICP. Those are static in the sensor frame and bias every
  registration toward "no motion". Not quantified.
- **41 % of the record is stationary** and no zero-velocity update is applied, so
  the odometry accumulates jitter while parked.