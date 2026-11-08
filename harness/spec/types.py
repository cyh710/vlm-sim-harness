"""Core data contracts and enums (design doc v0.3 section 2, runtime spec v0.2).

Pure dataclasses; tuples instead of ndarray so everything stays hashable and
deterministic without numpy in the contract layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional


class Status(str, Enum):
    REACHED = "reached"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    IK_FAILED = "ik_failed"
    UNSAFE = "unsafe"
    NOT_IMPLEMENTED = "not_implemented"


class Verdict(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    UNSAFE = "unsafe"


class ExitReason(str, Enum):
    SUCCESS = "SUCCESS"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SAFETY_STOP = "SAFETY_STOP"


class CallKind(str, Enum):
    QUERY = "query"
    ACTION = "action"
    CONTROL = "control"


class CallState(str, Enum):
    """Lifecycle of one call_id used to enforce the no-blind-resend invariant."""

    PENDING = "pending"          # proposed by planner, not yet validated
    DISPATCHED = "dispatched"    # sent to executor (physical side effects may occur)
    SETTLED = "settled"          # outcome known (verified or explicitly failed)
    UNKNOWN = "unknown"          # outcome not knowable (timeout); MUST NOT be re-sent


class Strategy(str, Enum):
    R0_EVIDENCE = "R0"           # refresh evidence / re-verify
    R1_LOCAL_FIX = "R1"          # re-locate, adjust grasp point, back through S3
    R2_REPLAN = "R2"             # change approach / intermediate subgoal
    R3_TERMINATE = "R3"          # controlled stop, report facts


class ReasonCode:
    """Closed vocabulary (design doc appendix A + patch P7).

    Extension codes are marked; everything else is a physical-action failure.
    """
    # perception
    PERCEPTION_STALE = "PERCEPTION_STALE"
    OBJECT_OCCLUDED = "OBJECT_OCCLUDED"
    OBJECT_NOT_VISIBLE = "OBJECT_NOT_VISIBLE"
    # grasp/manipulation
    GRASP_SLIPPED = "GRASP_SLIPPED"
    NOT_HOLDING = "NOT_HOLDING"
    NO_CONTACT = "NO_CONTACT"
    DROPPED = "DROPPED"
    TOPPLED = "TOPPLED"
    # planning/motion
    IK_UNREACHABLE = "IK_UNREACHABLE"
    PLAN_BLOCKED = "PLAN_BLOCKED"
    OUT_OF_TOLERANCE = "OUT_OF_TOLERANCE"
    # execution
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    HARDWARE_FAULT = "HARDWARE_FAULT"
    COLLISION_RISK = "COLLISION_RISK"
    # extension (P7)
    LIFT_NOT_ENOUGH = "LIFT_NOT_ENOUGH"
    PROXY_REGION_MISMATCH = "PROXY_REGION_MISMATCH"


class PlannerProtocolError(Exception):
    """LLM/scripted planner produced structurally invalid output."""


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def distance_to(self, other: "Pose") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2
                + (self.z - other.z) ** 2) ** 0.5


@dataclass(frozen=True)
class ObjInstance:
    id: str
    label: str
    view: str                       # "front" | "wrist"
    pose: Pose
    sigma_m: float
    size: tuple[float, float, float]
    visible_ratio: float
    quality: str                    # ok | low_confidence | ambiguous_yaw
    source: str                     # "cad" | "pca"
    last_seen_t: float


@dataclass(frozen=True)
class WorldState:
    version: int
    timestamp: float
    ee_pose: Pose
    gripper_open: float             # 0=fully closed, 1=fully open
    joints: dict[str, float]        # named joints (mock); real adapter maps URDF names
    objects: tuple[ObjInstance, ...]
    unsafe_flag: bool = False       # raised by adapter-level hazard detection
    evidence_refs: tuple[str, ...] = ()

    def object_by_id(self, object_id: str) -> Optional[ObjInstance]:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    kind: str                       # query | action | control (CallKind value)
    name: str
    args: dict
    world_version: int              # world version the planner based this call on
    expected_effects: tuple[dict, ...] = ()   # untrusted planner claim; audit only
    timeout_ms: int = 5000

    def validated(self) -> "ToolCall":
        return self


@dataclass(frozen=True)
class ExecutionResult:
    call_id: str
    status: Status
    feedback: dict                  # may contain executor "claimed" report (untrusted)
    evidence_refs: tuple[str, ...] = ()
    elapsed_ms: int = 0


@dataclass(frozen=True)
class CheckResult:
    effect: str                     # human-readable effect summary
    passed: Optional[bool]          # True / False / None(=unknown, evidence insufficient)
    value: Any
    threshold: Any
    signal: str                     # "vision" | "proprio" | "derived"
    counter_evidence: bool = False  # explicit counter-evidence found (not just absence)
    reason_code: Optional[str] = None


@dataclass(frozen=True)
class Evidence:
    call_id: str
    action: str
    args: dict
    expected_effects: tuple[dict, ...]
    before: WorldState
    after: WorldState
    metrics: dict
    claimed: dict                   # executor self-report; verifier MUST NOT read
    evidence_refs: tuple[str, ...] = ()
    timeout_exceeded: bool = False


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    reason_code: Optional[str]
    observed_effects: tuple[dict, ...]
    checks: tuple[CheckResult, ...]
    layer: str                      # "geometric" | "semantic"
    latency_s: float = 0.0


@dataclass(frozen=True)
class RecoveryDecision:
    strategy: Strategy
    reason_code: str
    next_state: str                 # "S1" | "S3" | "S8" | "EXIT"


@dataclass(frozen=True)
class ProgressDelta:
    verified_effects: tuple[dict, ...]
    source_call_id: str
    evidence_refs: tuple[str, ...]


@dataclass
class AttemptRecord:
    """Mutable by design: state transitions DISPATCHED -> SETTLED/UNKNOWN."""
    attempt_id: str
    call_id: str
    name: str
    args: dict
    state: CallState
    status: Optional[Status] = None
    started_at: float = 0.0
    ended_at: Optional[float] = None


@dataclass
class EpisodeResult:
    exit_reason: ExitReason
    online_claim_success: bool
    judge_success: Optional[bool] = None    # filled only by offline judge, never runtime
    physical_actions: int = 0
    grasp_attempts: int = 0
    queries: int = 0
    llm_calls: int = 0
    replans: int = 0
    inconclusive: int = 0
    recovery_attempts: int = 0
    recovery_success: int = 0
    wall_time_s: float = 0.0
    progress_entries: int = 0
    trace_events: int = 0

    def to_result_dict(self, run_id: str, episode_id: str, config: dict) -> dict:
        return {
            "run_id": run_id,
            "episode_id": episode_id,
            **config,
            "online_claim_success": self.online_claim_success,
            "judge_success": self.judge_success,
            "exit_reason": self.exit_reason.value,
            "physical_actions": self.physical_actions,
            "grasp_attempts": self.grasp_attempts,
            "queries": self.queries,
            "llm_calls": self.llm_calls,
            "replans": self.replans,
            "inconclusive": self.inconclusive,
            "recovery_attempts": self.recovery_attempts,
            "recovery_success": self.recovery_success,
            "wall_time_s": round(self.wall_time_s, 3),
            "progress_entries": self.progress_entries,
            "trace_events": self.trace_events,
        }
