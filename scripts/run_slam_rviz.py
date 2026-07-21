#!/usr/bin/env python3
"""Command-line runner for replaying offline SLAM results in RViz."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slam_icp.rviz_replay import SlamReplayNode


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a completed offline SLAM run in RViz"
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Completed SLAM output directory, "
            "for example outputs/test1_slam"
        ),
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Replay speed multiplier; default: 1.0",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart the trajectory automatically after it finishes",
    )

    parser.add_argument(
        "--rviz",
        action="store_true",
        help="Open RViz automatically using config/rviz/slam_replay.rviz",
    )

    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    arguments = parse_arguments()

    if arguments.rate <= 0.0:
        print(
            "RViz replay failed: --rate must be greater than zero",
            file=sys.stderr,
        )
        return 1

    rclpy.init()

    node: SlamReplayNode | None = None
    rviz_process: subprocess.Popen | None = None

    try:
        node = SlamReplayNode(
            run_directory=arguments.run_dir,
            replay_rate=arguments.rate,
            repeat=arguments.loop,
        )

        if arguments.rviz:
            rviz_configuration = (
                repository_root()
                / "config"
                / "rviz"
                / "slam_replay.rviz"
            )

            if not rviz_configuration.exists():
                raise FileNotFoundError(
                    "RViz configuration was not found: "
                    f"{rviz_configuration}"
                )

            rviz_process = subprocess.Popen(
                [
                    "rviz2",
                    "-d",
                    str(rviz_configuration),
                ]
            )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(
            f"RViz replay failed: {error}",
            file=sys.stderr,
        )
        return 1

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        if rviz_process is not None:
            rviz_process.terminate()

            try:
                rviz_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                rviz_process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
