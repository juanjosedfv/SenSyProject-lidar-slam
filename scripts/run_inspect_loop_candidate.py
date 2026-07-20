#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from slam_icp.loop_closure import (
    crop_around_location,
    filter_points_by_range,
    load_keyframe_by_id,
    load_nearest_pointcloud,
    plot_loop_candidate,
    pointcloud2_to_xyz,
    transform_points,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and compare the LiDAR scans "
            "for one loop-closure candidate."
        )
    )

    parser.add_argument(
        "--keyframes",
        required=True,
        help="Path to keyframes.csv",
    )

    parser.add_argument(
        "--lidar-bag",
        required=True,
        help="Path to the raw LiDAR rosbag",
    )

    parser.add_argument(
        "--lidar-topic",
        default="/ouster/points",
        help="PointCloud2 topic in the rosbag",
    )

    parser.add_argument(
        "--source-id",
        type=int,
        required=True,
        help="Source keyframe ID",
    )

    parser.add_argument(
        "--target-id",
        type=int,
        required=True,
        help="Target keyframe ID",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for loop inspection outputs",
    )

    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed LiDAR-to-pose "
            "timestamp difference in seconds"
        ),
    )

    parser.add_argument(
        "--min-range",
        type=float,
        default=1.0,
        help="Minimum LiDAR range in metres",
    )

    parser.add_argument(
        "--max-range",
        type=float,
        default=50.0,
        help="Maximum LiDAR range in metres",
    )

    parser.add_argument(
        "--crop-radius",
        type=float,
        default=30.0,
        help=(
            "Horizontal radius retained around "
            "the loop location, in metres"
        ),
    )

    parser.add_argument(
        "--plot-stride",
        type=int,
        default=10,
        help=(
            "Plot every Nth point to keep the "
            "image readable"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    output_directory = Path(
        arguments.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_keyframe = load_keyframe_by_id(
        arguments.keyframes,
        arguments.source_id,
    )

    target_keyframe = load_keyframe_by_id(
        arguments.keyframes,
        arguments.target_id,
    )

    source_message, source_dt = (
        load_nearest_pointcloud(
            bag_path=arguments.lidar_bag,
            topic_name=arguments.lidar_topic,
            target_timestamp=(
                source_keyframe["timestamp"]
            ),
            maximum_time_difference_s=(
                arguments.max_dt
            ),
        )
    )

    target_message, target_dt = (
        load_nearest_pointcloud(
            bag_path=arguments.lidar_bag,
            topic_name=arguments.lidar_topic,
            target_timestamp=(
                target_keyframe["timestamp"]
            ),
            maximum_time_difference_s=(
                arguments.max_dt
            ),
        )
    )

    source_points = pointcloud2_to_xyz(
        source_message
    )

    target_points = pointcloud2_to_xyz(
        target_message
    )

    source_points = filter_points_by_range(
        source_points,
        minimum_range_m=arguments.min_range,
        maximum_range_m=arguments.max_range,
    )

    target_points = filter_points_by_range(
        target_points,
        minimum_range_m=arguments.min_range,
        maximum_range_m=arguments.max_range,
    )

    source_world = transform_points(
        source_points,
        source_keyframe,
    )

    target_world = transform_points(
        target_points,
        target_keyframe,
    )

    loop_centre_x = (
        source_keyframe["x"]
        + target_keyframe["x"]
    ) / 2.0

    loop_centre_y = (
        source_keyframe["y"]
        + target_keyframe["y"]
    ) / 2.0

    source_crop = crop_around_location(
        source_world,
        centre_x=loop_centre_x,
        centre_y=loop_centre_y,
        radius_m=arguments.crop_radius,
    )

    target_crop = crop_around_location(
        target_world,
        centre_x=loop_centre_x,
        centre_y=loop_centre_y,
        radius_m=arguments.crop_radius,
    )

    output_plot = (
        output_directory
        / "plots"
        / (
            f"loop_candidate_"
            f"{arguments.source_id}_"
            f"{arguments.target_id}.png"
        )
    )

    plot_loop_candidate(
        path=output_plot,
        source_points=source_crop,
        target_points=target_crop,
        source_keyframe=source_keyframe,
        target_keyframe=target_keyframe,
        plot_stride=arguments.plot_stride,
    )

    print("Loop-candidate inspection completed")

    print(
        f"Source keyframe: "
        f"{arguments.source_id}"
    )

    print(
        f"Target keyframe: "
        f"{arguments.target_id}"
    )

    print(
        f"Source scan-pose dt: "
        f"{source_dt:.6f} s"
    )

    print(
        f"Target scan-pose dt: "
        f"{target_dt:.6f} s"
    )

    print(
        f"Source filtered points: "
        f"{len(source_points)}"
    )

    print(
        f"Target filtered points: "
        f"{len(target_points)}"
    )

    print(
        f"Source cropped points: "
        f"{len(source_crop)}"
    )

    print(
        f"Target cropped points: "
        f"{len(target_crop)}"
    )

    print(f"Plot: {output_plot}")


if __name__ == "__main__":
    main()