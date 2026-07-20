#!/usr/bin/env python3
"""Command-line entry point for the complete offline SLAM pipeline."""

from __future__ import annotations

import argparse
import sys

from slam_icp.config import SUPPORTED_BACKENDS, load_slam_config
from slam_icp.pipeline import run_slam


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backend-independent offline LiDAR SLAM")
    parser.add_argument("--config", required=True, help="SLAM YAML configuration")
    parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_BACKENDS),
        default=None,
        help="Override the backend selected in the configuration",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_slam_config(arguments.config, arguments.backend)
        result = run_slam(config)
    except Exception as error:
        print(f"SLAM failed: {error}", file=sys.stderr)
        return 1

    print("Offline SLAM completed")
    print(f"Backend: {result.backend}")
    print(f"Poses: {result.pose_count}")
    print(f"Keyframes: {result.keyframe_count}")
    print(f"Candidates: {result.candidate_count}")
    print(f"Accepted loops: {result.accepted_loop_count}")
    print(f"Corrected trajectory: {result.corrected_trajectory_path}")
    print(f"Map: {result.map_path if result.map_path is not None else 'disabled'}")
    print(f"Run summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
