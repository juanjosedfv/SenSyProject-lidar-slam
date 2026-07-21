# LIDAR-based Positioning

Implementation for the Sensor Systems project: LIDAR odometry / mapping for XTrack.

Defineded LIDAR tasks:

- simple baseline: LIDAR odometry with scan-scan or scan-map matching
- advanced version: SLAM
- outputs: 2D trajectory in Lat/Lon, velocity in NED, a 3D point cloud map, and error plots against corrected GNSS using metrics such as RMSE


## Goals

Produce:

- `trajectory_estimate.csv`: estimated 2D trajectory in latitude/longitude and local NED/ENU coordinates
- `velocity_ned.csv`: estimated velocity in NED frame
- `map.pcd`: accumulated 3D point cloud map
- evaluation plots comparing estimated trajectory against corrected GNSS ground truth

## Data Layout

The checked-in config points at the exported ROS bag data:

```text
data/
├── rosbag/
│   ├── metadata.yaml
│   ├── rosbag_0.db3
│   └── rosbag_data/
│       └── laz_clouds/
└── xtrack_gnss_corrected/
    └── xtrack_global_position_t12.csv
```

Paths and runtime limits can be configured in `config/config.yaml`.

## Setup

Use the course/project conda environment:

```bash
conda activate sensor_system
python -m pip install -r requirements.txt
```

For LAZ support, `laspy[lazrs]` is used. ICP nearest-neighbor search uses SciPy.

## Run

```bash
python -m src.main --config config/config.yaml
```

Direct script execution also works:

```bash
python src/main.py --config config/config.yaml
```

The default config uses a small slice of the 6008 Ouster scans:

- `max_scans: 120`
- `scan_stride: 5`
- `start_index: 300`

This is intentional for fast tuning. Set `max_scans: null` and `scan_stride: 1` for a full run after the parameters look sane.

Use CLI overrides for quick experiments:

```bash
python src/main.py --config config/config.yaml --max-scans 20 --scan-stride 5 --start-index 300
```

## Outputs

The runner writes:

- `outputs/trajectory_estimate.csv`: timestamped LIDAR trajectory, aligned ENU, and Lat/Lon
- `outputs/velocity_ned.csv`: velocity in NED frame
- `outputs/map.pcd`: accumulated downsampled XYZ map
- `outputs/metrics.json`: scan count, GNSS overlap count, horizontal RMSE, ICP fitness/RMSE means, and path-length sanity diagnostics
- `outputs/plots/trajectory_vs_gnss.png`
- `outputs/plots/horizontal_error.png`

## Method

1. Validate the provided ROS bag at `data/rosbag/rosbag_0.db3` and count `/ouster/points` messages.
2. Load timestamped `.laz` scans exported from that ROS bag under `data/rosbag/rosbag_data/laz_clouds`.
3. Filter invalid/range-limited points and voxel-downsample each scan.
4. Estimate relative motion with point-to-point ICP.
5. Accumulate transformed scans into a downsampled 3D map.
6. Align the local LIDAR trajectory to corrected GNSS from `data/xtrack_gnss_corrected/xtrack_global_position_t12.csv`.
7. Export trajectory, velocity, map, plots, and metrics.

The corrected GNSS file is used only as ground truth for alignment/evaluation, not as an input to ICP.

## Current Limitation

The current point-to-point ICP implementation is a baseline and data-pipeline scaffold, not a final odometry result. If `trajectory_vs_gnss.png` shows the orange LIDAR trajectory collapsed into a tiny segment while the blue GNSS trajectory moves many meters, that is a real failure mode: the baseline ICP is stuck near identity.

Check `outputs/metrics.json`:

- `raw_lidar_path_m`: how far the unaligned LIDAR odometry thinks the platform moved
- `gnss_path_m`: how far corrected GNSS moved in the same timestamps
- `raw_lidar_to_gnss_path_ratio`: rough sanity ratio

A very small ratio means we need a stronger odometry backend or a better motion prior.

## Optional Toolboxes

The project slides suggest external SLAM/odometry repositories. Practical order for our data:

1. **KISS-ICP**: best first upgrade for pure LiDAR odometry.
2. **koide3 tools**: useful for stronger GICP/SLAM experiments, especially `small_gicp`, `fast_gicp`, or `glim`.
3. **LIO-SAM 6AXIS**: relevant for a full LiDAR-inertial-GNSS pipeline, but heavier because it expects ROS/catkin-style setup and sensor-specific configuration.
4. **GenZ-ICP / lidar_odometry**: interesting C++ comparison or stretch goal.
5. **ORB-SLAM3**: strong visual/visual-inertial SLAM, but less aligned with the LIDAR-based task unless we intentionally use RGB-D/camera data.

KISS-ICP failed to build in the current local Python 3.14 conda env. It is still the recommended next backend, but likely in a Python 3.11/3.12 env or ROS 2/Docker setup.
