"""Loading and validation for the full offline SLAM configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_BACKENDS = {"kiss_icp", "baseline_icp"}


@dataclass(frozen=True)
class SlamConfig:
    data: dict[str, Any]
    source_path: Path
    repository_root: Path

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.repository_root / path


def _require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' is required")
    return section


def _positive(section: dict[str, Any], key: str) -> None:
    if float(section[key]) <= 0.0:
        raise ValueError(f"Configuration value '{key}' must be positive")


def validate_slam_config(data: dict[str, Any]) -> None:
    for name in (
        "run", "inputs", "frames", "keyframes", "loop_search",
        "loop_registration", "loop_closure", "optimization", "mapping",
        "evaluation", "visualization", "outputs",
    ):
        _require_section(data, name)

    run = data["run"]
    if run.get("backend") not in SUPPORTED_BACKENDS:
        raise ValueError("run.backend must be kiss_icp or baseline_icp")
    if not str(run.get("name", "")).strip():
        raise ValueError("run.name is required")

    for section_name, keys in {
        "keyframes": ("translation_threshold_m", "yaw_threshold_deg", "time_threshold_s"),
        "loop_search": ("radius_m", "minimum_id_separation", "maximum_candidates"),
        "loop_registration": (
            "pose_tolerance_s", "maximum_range_m", "voxel_size_m",
            "maximum_correspondence_distance_m", "maximum_iterations",
        ),
        "optimization": (
            "odometry_translation_sigma_m", "odometry_yaw_sigma_deg",
            "loop_translation_sigma_m", "loop_yaw_sigma_deg",
            "maximum_evaluations", "gradient_tolerance",
        ),
    }.items():
        section = data[section_name]
        for key in keys:
            if key not in section:
                raise ValueError(f"Missing configuration value '{section_name}.{key}'")
            _positive(section, key)

    pairs = data["loop_closure"].get("accepted_pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("loop_closure.accepted_pairs must be a list")
    for pair in pairs:
        if not isinstance(pair, dict) or "source_id" not in pair or "target_id" not in pair:
            raise ValueError("Each accepted loop pair needs source_id and target_id")

    if data["evaluation"].get("scale_alignment", False):
        raise ValueError("Scale alignment is not supported by the validated workflow")


def load_slam_config(
    path: str | Path,
    backend_override: str | None = None,
) -> SlamConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"SLAM configuration does not exist: {source_path}")
    loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("SLAM configuration must contain a YAML mapping")
    data = deepcopy(loaded)
    if backend_override is not None:
        if backend_override not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend_override}")
        _require_section(data, "run")["backend"] = backend_override
    validate_slam_config(data)
    repository_root = source_path.parent.parent
    return SlamConfig(data=data, source_path=source_path, repository_root=repository_root)
