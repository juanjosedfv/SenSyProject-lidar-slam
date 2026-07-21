"""Backend-independent in-memory orchestration for offline LiDAR SLAM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from slam_icp.backends import LOADERS
from slam_icp.config import SlamConfig, load_slam_config, validate_slam_config
from slam_icp.io import write_json
from slam_icp.models import (
    GraphEdge,
    GraphNode,
    Keyframe,
    LoopCandidate,
    LoopConstraint,
    PoseGraph,
    SlamRunResult,
)
from slam_icp.pose_graph import (
    apply_keyframe_corrections,
    build_keyframe_corrections,
    calculate_correction_summary,
    plot_full_trajectory_correction,
    write_corrected_trajectory_csv,
)
from slam_icp.keyframes import (
    plot_keyframes,
    select_keyframes,
    write_keyframe_summary,
    write_keyframes_csv,
)
from slam_icp.loop_closure import (
    find_loop_candidates,
    plot_loop_candidates,
    select_loop_registrations,
    write_loop_candidates_csv,
)
from slam_icp.loop_closure import register_loop_pair
from slam_icp.pose_graph import (
    calculate_node_corrections,
    optimize_pose_graph,
    plot_optimized_graph,
    write_optimized_nodes_csv,
)
from slam_icp.pose_graph import (
    build_odometry_edges,
    calculate_reconstruction_errors,
    plot_initial_pose_graph,
    reconstruct_graph_trajectory,
    write_edges_csv,
    write_graph_summary,
    write_nodes_csv,
    write_reconstructed_csv,
)
from slam_icp.outputs import evaluate_slam
from slam_icp.outputs import build_slam_map


def _path(config: SlamConfig, section: str, key: str, default: str | None = None) -> Path:
    value = config.data[section].get(key, default)
    if value is None:
        raise ValueError(f"Missing path configuration: {section}.{key}")
    return config.path(value)


def _pose_dicts(poses) -> list[dict[str, Any]]:
    return [
        {
            "pose_index": index,
            "timestamp": pose.timestamp,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "qx": pose.qx,
            "qy": pose.qy,
            "qz": pose.qz,
            "qw": pose.qw,
            "backend": pose.backend,
            "parent_frame": pose.parent_frame,
            "child_frame": pose.child_frame,
        }
        for index, pose in enumerate(poses)
    ]


def _graph_node(keyframe: Keyframe) -> GraphNode:
    return GraphNode(
        keyframe_id=keyframe.keyframe_id,
        pose_index=keyframe.pose_index,
        timestamp=keyframe.timestamp,
        x=keyframe.x,
        y=keyframe.y,
        yaw_rad=math.radians(keyframe.yaw_deg),
        backend=keyframe.backend,
        parent_frame=keyframe.parent_frame,
        child_frame=keyframe.child_frame,
        fixed=keyframe.keyframe_id == 0,
    )


def _optimization_summary(
    original_nodes: list[dict],
    optimized_nodes: list[dict],
    odometry_edges: list[dict],
    constraints: list[dict],
    solver_result,
    initial_cost: float,
    settings: dict,
) -> dict:
    return {
        "node_count": len(original_nodes),
        "odometry_edge_count": len(odometry_edges),
        "loop_edge_count": len(constraints),
        "loop_pairs": [
            {"source_id": item["source_id"], "target_id": item["target_id"]}
            for item in constraints
        ],
        "solver_success": bool(solver_result.success) if solver_result is not None else True,
        "solver_status": int(solver_result.status) if solver_result is not None else 0,
        "solver_message": (
            str(solver_result.message)
            if solver_result is not None
            else "No loop constraints; optimization was not required"
        ),
        "solver_function_evaluations": int(solver_result.nfev) if solver_result is not None else 0,
        "solver_optimality": float(solver_result.optimality) if solver_result is not None else 0.0,
        "initial_cost": float(initial_cost),
        "final_cost": float(solver_result.cost) if solver_result is not None else 0.0,
        **calculate_node_corrections(original_nodes, optimized_nodes),
        "settings": settings,
    }


def _validate_frames(config: SlamConfig, metadata: dict[str, Any]) -> None:
    frames = config.data["frames"]
    if frames["parent_frame"] != metadata["parent_frame"]:
        raise ValueError("Configured parent frame does not match the trajectory")
    if frames["child_frame"] != metadata["child_frame"]:
        raise ValueError("Configured child frame does not match the trajectory")


def _loaded_config(
    config: SlamConfig | str | Path,
    backend_override: str | None,
) -> SlamConfig:
    if isinstance(config, (str, Path)):
        return load_slam_config(config, backend_override)
    if backend_override is None:
        return config
    data = {**config.data, "run": {**config.data["run"], "backend": backend_override}}
    validate_slam_config(data)
    return SlamConfig(
        data=data,
        source_path=config.source_path,
        repository_root=config.repository_root,
    )


def run_slam(
    config: SlamConfig | str | Path,
    backend_override: str | None = None,
) -> SlamRunResult:
    """Run the existing offline SLAM stages without intermediate reloads."""

    loaded = _loaded_config(config, backend_override)
    data = loaded.data
    backend = data["run"]["backend"]
    backend_label = "KISS-ICP" if backend == "kiss_icp" else "baseline_icp"
    trajectory_path = _path(loaded, "inputs", "trajectory_csv")
    backend_trajectory = LOADERS[backend](trajectory_path)
    _validate_frames(loaded, backend_trajectory.metadata)
    poses = _pose_dicts(backend_trajectory.poses)

    output_dir = _path(loaded, "outputs", "output_dir")
    checkpoints_dir = _path(loaded, "outputs", "checkpoints_dir")
    plots_dir = _path(loaded, "outputs", "plots_dir")
    corrected_path = _path(loaded, "outputs", "corrected_trajectory")
    map_path = _path(loaded, "outputs", "corrected_map")
    summary_path = _path(loaded, "outputs", "run_summary")
    metrics_path = _path(loaded, "outputs", "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_dir = summary_path.parent
    internal_dir.mkdir(parents=True, exist_ok=True)
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    effective_config_path = internal_dir / "effective_config.yaml"
    effective_config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    backend_metadata_path = write_json(internal_dir / "backend_metadata.json", backend_trajectory.metadata)

    keyframe_cfg = data["keyframes"]
    keyframe_values = select_keyframes(
        poses,
        translation_threshold_m=float(keyframe_cfg["translation_threshold_m"]),
        yaw_threshold_deg=float(keyframe_cfg["yaw_threshold_deg"]),
        time_threshold_s=float(keyframe_cfg["time_threshold_s"]),
    )
    keyframes = [Keyframe.from_dict(value) for value in keyframe_values]
    keyframe_dicts = [item.to_dict() for item in keyframes]
    nodes = [_graph_node(item) for item in keyframes]
    node_dicts = [item.to_dict() for item in nodes]
    edge_values = build_odometry_edges(node_dicts)
    edges = [GraphEdge.from_dict(value) for value in edge_values]
    edge_dicts = [item.to_dict() for item in edges]
    graph = PoseGraph(nodes=nodes, odometry_edges=edges)

    search_cfg = data["loop_search"]
    candidate_values = find_loop_candidates(
        node_dicts,
        search_radius_m=float(search_cfg["radius_m"]),
        minimum_time_separation_s=float(search_cfg["minimum_time_separation_s"]),
        minimum_id_separation=int(search_cfg["minimum_id_separation"]),
        maximum_candidates=int(search_cfg["maximum_candidates"]),
    )
    candidates = [LoopCandidate.from_dict(value) for value in candidate_values]

    save_checkpoints = bool(data["run"].get("save_checkpoints", False))
    visualization = data["visualization"]
    visualization_enabled = bool(visualization.get("enabled", False))
    diagnostic_plots_enabled = visualization_enabled and bool(
        visualization.get("generate_diagnostic_plots", False)
    )
    plot_paths: list[str] = []

    if save_checkpoints:
        reconstructed = reconstruct_graph_trajectory(node_dicts, edge_dicts)
        errors = calculate_reconstruction_errors(node_dicts, reconstructed)
        write_keyframes_csv(checkpoints_dir / "keyframes.csv", keyframe_dicts)
        write_keyframe_summary(checkpoints_dir / "keyframe_summary.json", len(poses), keyframe_dicts)
        write_nodes_csv(checkpoints_dir / "pose_graph_nodes.csv", node_dicts)
        write_edges_csv(checkpoints_dir / "pose_graph_edges.csv", edge_dicts)
        write_reconstructed_csv(checkpoints_dir / "pose_graph_reconstructed.csv", reconstructed)
        write_graph_summary(checkpoints_dir / "pose_graph_summary.json", node_dicts, edge_dicts, errors)
        write_loop_candidates_csv(checkpoints_dir / "loop_candidates.csv", candidate_values)

    if diagnostic_plots_enabled:
        keyframe_plot = plots_dir / "keyframes_on_trajectory.png"
        graph_plot = plots_dir / "pose_graph_initial.png"
        candidate_plot = plots_dir / "loop_candidates.png"
        plot_keyframes(keyframe_plot, poses, keyframe_dicts)
        plot_initial_pose_graph(graph_plot, node_dicts)
        plot_loop_candidates(
            candidate_plot,
            node_dicts,
            candidate_values,
            trajectory_label=f"{backend_label} keyframe trajectory",
        )
        plot_paths.extend(map(str, (keyframe_plot, graph_plot, candidate_plot)))

    registration_cfg = data["loop_registration"]
    loop_closure_cfg = data["loop_closure"]
    loop_closure_mode = loop_closure_cfg["mode"]

    def register_pair(source_id: int, target_id: int) -> dict:
        return register_loop_pair(
            keyframes=keyframe_dicts,
            lidar_bag_path=_path(loaded, "inputs", "lidar_bag"),
            lidar_topic=data["inputs"]["lidar_topic"],
            source_id=source_id,
            target_id=target_id,
            maximum_time_difference_s=float(registration_cfg["pose_tolerance_s"]),
            minimum_range_m=float(registration_cfg["minimum_range_m"]),
            maximum_range_m=float(registration_cfg["maximum_range_m"]),
            voxel_size_m=float(registration_cfg["voxel_size_m"]),
            maximum_correspondence_distance_m=float(registration_cfg["maximum_correspondence_distance_m"]),
            maximum_iterations=int(registration_cfg["maximum_iterations"]),
            output_directory=checkpoints_dir,
            plot_stride=int(visualization["plot_stride"]),
            generate_plots=diagnostic_plots_enabled and loop_closure_mode == "manual",
            write_constraint=save_checkpoints and loop_closure_mode == "manual",
        )

    selected_registrations, loop_decisions = select_loop_registrations(
        mode=loop_closure_mode,
        candidates=candidate_values,
        manual_pairs=loop_closure_cfg.get("manual_pairs", []),
        register_pair=register_pair,
        minimum_fitness=float(loop_closure_cfg.get("minimum_fitness", 0.30)),
        maximum_inlier_rmse_m=float(
            loop_closure_cfg.get("maximum_inlier_rmse_m", 0.50)
        ),
        maximum_translation_correction_m=float(
            loop_closure_cfg.get("maximum_translation_correction_m", 5.0)
        ),
        maximum_yaw_correction_deg=float(
            loop_closure_cfg.get("maximum_yaw_correction_deg", 20.0)
        ),
        maximum_accepted_loops=int(
            loop_closure_cfg.get("maximum_accepted_loops", 1)
        ),
    )
    loop_decisions_path = write_json(internal_dir / "loop_decisions.json", loop_decisions)
    if save_checkpoints and loop_closure_mode == "auto":
        for value in selected_registrations:
            constraint_path = write_json(
                checkpoints_dir
                / f"loop_constraint_{value['source_id']}_{value['target_id']}.json",
                value,
            )
            value["constraint_path"] = str(constraint_path)

    constraints: list[LoopConstraint] = []
    for value in selected_registrations:
        constraint = LoopConstraint.from_dict(value)
        constraints.append(constraint)
        for key in ("before_plot", "after_plot"):
            if value.get(key):
                plot_paths.append(value[key])
    graph.loop_constraints = constraints
    constraint_dicts = [item.to_dict() for item in constraints]
    accepted_constraints_path = write_json(internal_dir / "accepted_loop_constraints.json", constraint_dicts)

    optimization_cfg = data["optimization"]
    settings = {
        "odometry_translation_sigma_m": float(optimization_cfg["odometry_translation_sigma_m"]),
        "odometry_yaw_sigma_deg": float(optimization_cfg["odometry_yaw_sigma_deg"]),
        "loop_translation_sigma_m": float(optimization_cfg["loop_translation_sigma_m"]),
        "loop_yaw_sigma_deg": float(optimization_cfg["loop_yaw_sigma_deg"]),
        "maximum_evaluations": int(optimization_cfg["maximum_evaluations"]),
        "gradient_tolerance": float(optimization_cfg["gradient_tolerance"]),
    }
    if constraint_dicts:
        optimized_nodes, solver_result, initial_cost = optimize_pose_graph(
            nodes=node_dicts,
            odometry_edges=edge_dicts,
            loop_constraint=constraint_dicts,
            **settings,
        )
    else:
        optimized_nodes = [dict(item) for item in node_dicts]
        solver_result = None
        initial_cost = 0.0
    optimization_summary = _optimization_summary(
        node_dicts,
        optimized_nodes,
        edge_dicts,
        constraint_dicts,
        solver_result,
        initial_cost,
        settings,
    )
    write_json(internal_dir / "pose_graph_optimization_summary.json", optimization_summary)
    if save_checkpoints:
        write_optimized_nodes_csv(checkpoints_dir / "pose_graph_optimized.csv", node_dicts, optimized_nodes)
    if diagnostic_plots_enabled and constraint_dicts:
        optimized_plot = plots_dir / "pose_graph_optimized.png"
        plot_optimized_graph(optimized_plot, node_dicts, optimized_nodes, constraint_dicts[0])
        plot_paths.append(str(optimized_plot))

    corrections = build_keyframe_corrections(keyframe_dicts, node_dicts, optimized_nodes)
    corrected_poses = apply_keyframe_corrections(poses, corrections)
    correction_summary = calculate_correction_summary(corrected_poses, corrections)
    write_corrected_trajectory_csv(corrected_path, corrected_poses)
    write_json(internal_dir / "trajectory_slam_summary.json", correction_summary)
    if diagnostic_plots_enabled:
        correction_plot = plots_dir / "trajectory_full_before_after.png"
        plot_full_trajectory_correction(
            correction_plot,
            poses,
            corrected_poses,
            original_label=f"Original {backend_label} trajectory",
        )
        plot_paths.append(str(correction_plot))

    map_summary = None
    if bool(data["mapping"].get("enabled", True)):
        mapping_cfg = data["mapping"]
        map_summary = build_slam_map(
            trajectory_path=corrected_path,
            lidar_bag_path=_path(loaded, "inputs", "lidar_bag"),
            lidar_topic=data["inputs"]["lidar_topic"],
            output_pcd_path=map_path,
            scan_stride=int(mapping_cfg["scan_stride"]),
            voxel_size_m=float(mapping_cfg["voxel_size_m"]),
            minimum_range_m=float(mapping_cfg["minimum_range_m"]),
            maximum_range_m=float(mapping_cfg["maximum_range_m"]),
            pose_tolerance_s=float(mapping_cfg["pose_tolerance_s"]),
            level_enabled=bool(mapping_cfg["leveling_enabled"]),
            level_reference_trajectory_path=_path(loaded, "mapping", "leveling_reference_trajectory"),
            trajectory_poses=corrected_poses,
            level_reference_poses=poses,
            summary_path=internal_dir / "map_slam_summary.json",
        )

    evaluation_summary = None
    if bool(data["evaluation"].get("enabled", False)):
        evaluation_cfg = data["evaluation"]
        evaluation_summary = evaluate_slam(
            original_trajectory_path=trajectory_path,
            slam_trajectory_path=corrected_path,
            gnss_csv_path=_path(loaded, "evaluation", "gnss_csv"),
            gnss_time_offset_s=float(evaluation_cfg["gnss_time_offset_s"]),
            level_enabled=bool(evaluation_cfg["leveling_enabled"]),
            output_directory=internal_dir / "evaluation",
            original_poses=poses,
            slam_poses=corrected_poses,
            write_detailed_csv=save_checkpoints,
            generate_plots=visualization_enabled and bool(visualization.get("generate_rmse_plot", True)),
            original_label=backend_label,
            plot_directory=plots_dir / "rmse",
            trajectory_plot_name="trajectory_slam_vs_gnss.png",
            error_plot_name="RMSE_SLAM_vs_GNSS.png",
        )
        write_json(metrics_path, evaluation_summary)
        for key in ("trajectory_plot", "error_plot"):
            if evaluation_summary.get(key):
                plot_paths.append(evaluation_summary[key])

    if (
        visualization_enabled
        and bool(visualization.get("generate_map_plots", False))
        and map_summary is not None
    ):
        from slam_icp.outputs import generate_map_plots

        pair = constraint_dicts[0] if constraint_dicts else None
        plot_summary = generate_map_plots(
            kiss_map_path=_path(loaded, "inputs", "reference_map"),
            slam_map_path=map_path,
            kiss_trajectory_path=trajectory_path,
            slam_trajectory_path=corrected_path,
            gnss_csv_path=_path(loaded, "evaluation", "gnss_csv"),
            gnss_time_offset_s=float(data["evaluation"]["gnss_time_offset_s"]),
            keyframes_path=checkpoints_dir / "keyframes.csv",
            level_reference_trajectory_path=_path(loaded, "mapping", "leveling_reference_trajectory"),
            source_id=(None if pair is None else int(pair["source_id"])),
            target_id=(None if pair is None else int(pair["target_id"])),
            output_directory=plots_dir / "maps",
            loop_radius_m=float(visualization["loop_radius_m"]),
            plot_stride=int(visualization["plot_stride"]),
            minimum_z_m=visualization.get("z_min"),
            maximum_z_m=visualization.get("z_max"),
            kiss_poses=poses,
            slam_poses=corrected_poses,
            keyframes=keyframe_dicts,
            level_reference_poses=poses,
            selected_outputs=visualization.get("map_plot_outputs"),
            map_plot_summary_path=internal_dir / "map_plot_summary.json",
        )
        plot_paths.extend(plot_summary["outputs"].values())

    summary = {
        "run_name": data["run"]["name"],
        "selected_backend": backend,
        "input_trajectory_path": str(trajectory_path),
        "pose_count": len(poses),
        "keyframe_count": len(keyframes),
        "odometry_edge_count": len(edges),
        "candidate_count": len(candidates),
        "loop_closure_mode": loop_closure_mode,
        "accepted_loop_count": len(constraints),
        "accepted_loop_ids": [
            {"source_id": item.source_id, "target_id": item.target_id}
            for item in constraints
        ],
        "solver": optimization_summary,
        "corrected_trajectory_path": str(corrected_path),
        "map_path": str(map_path) if map_summary is not None else None,
        "map_point_count": map_summary["final_voxel_count"] if map_summary else None,
        "mapping": map_summary,
        "evaluation_metrics": evaluation_summary,
        "plot_paths": plot_paths,
        "effective_configuration_path": str(effective_config_path),
        "backend_metadata_path": str(backend_metadata_path),
        "accepted_constraints_path": str(accepted_constraints_path),
        "loop_decisions_path": str(loop_decisions_path),
        "checkpoints_saved": save_checkpoints,
    }
    write_json(summary_path, summary)
    return SlamRunResult(
        backend=backend,
        pose_count=len(poses),
        keyframe_count=len(keyframes),
        candidate_count=len(candidates),
        accepted_loop_count=len(constraints),
        corrected_trajectory_path=corrected_path,
        map_path=map_path if map_summary is not None else None,
        summary_path=summary_path,
        metrics=evaluation_summary,
        summary=summary,
    )
