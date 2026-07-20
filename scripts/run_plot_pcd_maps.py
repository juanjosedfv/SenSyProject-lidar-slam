#!/usr/bin/env python3
"""Generate configured KISS/SLAM PCD and route plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slam_icp.config import load_slam_config
from slam_icp.outputs import generate_map_plots


DEFAULT_OUTPUTS = [
    "slam_map_topdown_height",
    "route_kiss_vs_gnss",
    "slam_map_with_route_and_gnss",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configured height-colored PCD maps and route plots."
    )
    parser.add_argument("--config", default="config/slam.yaml")
    parser.add_argument("--source-id", type=int, default=None)
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-stride", type=int, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    return parser.parse_args()


def _accepted_pair(config_data: dict, source_id: int | None, target_id: int | None) -> tuple[int, int]:
    if source_id is not None or target_id is not None:
        if source_id is None or target_id is None:
            raise ValueError("Both --source-id and --target-id are required when overriding the loop pair")
        return source_id, target_id
    pairs = config_data["loop_closure"].get("accepted_pairs", [])
    if not pairs:
        raise ValueError("The config has no accepted loop pair to plot")
    return int(pairs[0]["source_id"]), int(pairs[0]["target_id"])


def main() -> None:
    arguments = parse_arguments()
    config = load_slam_config(arguments.config)
    data = config.data
    source_id, target_id = _accepted_pair(data, arguments.source_id, arguments.target_id)
    plot_outputs = data["visualization"].get("map_plot_outputs") or DEFAULT_OUTPUTS
    plots_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir is not None
        else config.path(data["outputs"]["plots_dir"]) / "maps"
    )
    summary = generate_map_plots(
        kiss_map_path=config.path(data["inputs"]["reference_map"]),
        slam_map_path=config.path(data["outputs"]["corrected_map"]),
        kiss_trajectory_path=config.path(data["inputs"]["trajectory_csv"]),
        slam_trajectory_path=config.path(data["outputs"]["corrected_trajectory"]),
        gnss_csv_path=config.path(data["evaluation"]["gnss_csv"]),
        gnss_time_offset_s=float(data["evaluation"]["gnss_time_offset_s"]),
        keyframes_path=config.path(data["outputs"]["checkpoints_dir"]) / "keyframes.csv",
        level_reference_trajectory_path=config.path(data["mapping"]["leveling_reference_trajectory"]),
        source_id=source_id,
        target_id=target_id,
        output_directory=plots_dir,
        loop_radius_m=float(data["visualization"]["loop_radius_m"]),
        plot_stride=int(arguments.plot_stride or data["visualization"]["plot_stride"]),
        minimum_z_m=arguments.z_min if arguments.z_min is not None else data["visualization"].get("z_min"),
        maximum_z_m=arguments.z_max if arguments.z_max is not None else data["visualization"].get("z_max"),
        selected_outputs=plot_outputs,
    )
    print(f"Wrote {len(summary['outputs'])} plots")
    for output in summary["outputs"].values():
        print(output)
    print(summary["summary_path"])


if __name__ == "__main__":
    main()
