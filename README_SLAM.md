# Sensor Systems LiDAR SLAM

Offline LiDAR SLAM workspace with two ICP backends and one SLAM pipeline.

- `kiss_icp` is the default backend for the validated Test1 workflow.
- `baseline_icp` is the alternative ICP implementation.
- `slam_icp` contains the new SLAM pipeline built on existing ICP trajectories.

GNSS is used only for evaluation and plots. It is not used to estimate the SLAM trajectory.

## Structure

```text
config/      YAML configuration files
scripts/     command-line runners
src/         backend, SLAM, and shared processing code
data/        local datasets, not required in Git
outputs/     generated outputs, not required in Git
```

## Setup

Use the project environment and install Python dependencies:

```bash
conda activate sensor_system
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

For commands that read ROS 2 bags, source ROS first:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

ROS packages such as `rclpy`, `sensor_msgs`, and `rosbag2_py` must come from the ROS installation.

## Commands

Generate the KISS-ICP trajectory outputs:

```bash
python scripts/run_kiss_evaluation.py --config config/kiss_icp.yaml
```

Build the KISS-ICP map:

```bash
python scripts/run_kiss_map.py --config config/kiss_icp.yaml
```

Run the full SLAM pipeline from the existing backend trajectory:

```bash
python scripts/run_slam.py --config config/slam.yaml --backend kiss_icp
```

Generate the selected point-cloud and route plots:

```bash
python scripts/run_plot_pcd_maps.py --config config/slam.yaml
```

Run the alternative baseline ICP backend:

```bash
python scripts/run_baseline_icp.py --config config/baseline_icp.yaml
```

## Outputs

The normal SLAM run keeps the main deliverables visible in `outputs/test1_slam/`:

- `map_slam.pcd`
- `RMSE_SLAM_metrics.json`
- `plots/maps/slam_map_topdown_height.png`
- `plots/maps/route_slam_vs_gnss.png`
- `plots/maps/slam_map_with_route_and_gnss.png`
- `plots/rmse/RMSE_SLAM_vs_GNSS.png`

Run details and reproducibility files are kept in `outputs/test1_slam/_internal/`.
