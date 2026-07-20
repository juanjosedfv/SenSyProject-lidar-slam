"""Small typed records passed between offline SLAM stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Keyframe:
    keyframe_id: int
    pose_index: int
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw_deg: float
    backend: str
    parent_frame: str
    child_frame: str
    distance_from_last_m: float
    yaw_change_from_last_deg: float
    time_from_last_s: float
    selection_reason: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Keyframe":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphNode:
    keyframe_id: int
    pose_index: int
    timestamp: float
    x: float
    y: float
    yaw_rad: float
    backend: str
    parent_frame: str
    child_frame: str
    fixed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    edge_id: int
    source_id: int
    target_id: int
    edge_type: str
    dx: float
    dy: float
    dyaw_rad: float
    dyaw_deg: float
    translation_m: float
    time_difference_s: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphEdge":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopCandidate:
    candidate_id: int
    source_id: int
    target_id: int
    source_timestamp: float
    target_timestamp: float
    source_x: float
    source_y: float
    target_x: float
    target_y: float
    distance_m: float
    time_separation_s: float
    id_separation: int
    yaw_difference_deg: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LoopCandidate":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopConstraint:
    source_id: int
    target_id: int
    dx: float
    dy: float
    dyaw_rad: float
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LoopConstraint":
        required = {name: value[name] for name in ("source_id", "target_id", "dx", "dy", "dyaw_rad")}
        details = {key: item for key, item in value.items() if key not in required}
        return cls(**required, details=details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": "loop",
            "dx": self.dx,
            "dy": self.dy,
            "dyaw_rad": self.dyaw_rad,
            **self.details,
        }


@dataclass
class PoseGraph:
    nodes: list[GraphNode]
    odometry_edges: list[GraphEdge]
    loop_constraints: list[LoopConstraint] = field(default_factory=list)


@dataclass(frozen=True)
class SlamRunResult:
    backend: str
    pose_count: int
    keyframe_count: int
    candidate_count: int
    accepted_loop_count: int
    corrected_trajectory_path: Path
    map_path: Path | None
    summary_path: Path
    metrics: dict[str, Any] | None
    summary: dict[str, Any]
