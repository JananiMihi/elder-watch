from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pose import PoseLandmarks
from .utils import Event, now_ms


@dataclass
class DecisionConfig:
    min_confirm_frames: int = 3
    center_of_mass_y_thresh: float = 0.4
    torso_vertical_angle_deg_thresh: float = 30.0
    immobile_speed_thresh: float = 0.015
    immobile_min_frames: int = 90  # ~3s at 30 FPS


class TemporalDecision:
    """
    Start rule-based (proposal section 4.9) and later evolve into RF/classifier.
    This version is intentionally minimal so Step 1 can run.
    """

    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self._fall_votes = 0
        self._near_votes = 0
        self._immobile_frames = 0
        self._last_fall_frame = -1000  # Track last fall alert to avoid spam

    def update(
        self,
        *,
        yolo_fall_conf: float | None,
        pose: PoseLandmarks | None,
        features: dict[str, Any] | None = None,
    ) -> Event:
        # Multi-cue rule-based logic (proposal section 4.9) with temporal smoothing.
        conf = float(yolo_fall_conf or 0.0)
        f = features or {}

        torso_angle = float(f.get("torso_angle_from_vertical_deg", 0.0))
        com_y = float(f.get("com_y", 1.0))
        speed = float(f.get("com_speed", 0.0))

        pose_says_risky = False
        # com_y grows downward; smaller com_y = "higher in frame" (not always meaningful).
        # We still keep threshold configurable; the primary robust cue here is torso angle.
        if pose is not None:
            pose_says_risky = (torso_angle >= (90.0 - self.cfg.torso_vertical_angle_deg_thresh)) or (
                com_y <= self.cfg.center_of_mass_y_thresh
            )

        fall_vote = (conf >= 0.7) or pose_says_risky
        near_vote = pose is not None and (torso_angle >= (90.0 - self.cfg.torso_vertical_angle_deg_thresh / 2.0))

        if fall_vote:
            self._fall_votes += 1
        else:
            self._fall_votes = max(0, self._fall_votes - 1)

        if near_vote:
            self._near_votes += 1
        else:
            self._near_votes = max(0, self._near_votes - 1)

        if pose is not None and speed <= self.cfg.immobile_speed_thresh:
            self._immobile_frames += 1
        else:
            self._immobile_frames = 0

        if self._fall_votes >= self.cfg.min_confirm_frames:
            return Event(
                kind="fall",
                confidence=min(1.0, max(0.7, conf)),
                timestamp_ms=now_ms(),
                details={"votes": self._fall_votes, "features": f},
            )

        if self._immobile_frames >= self.cfg.immobile_min_frames:
            return Event(
                kind="immobile",
                confidence=0.8,
                timestamp_ms=now_ms(),
                details={"immobile_frames": self._immobile_frames, "features": f},
            )

        if self._near_votes >= self.cfg.min_confirm_frames:
            return Event(
                kind="near_fall",
                confidence=0.6,
                timestamp_ms=now_ms(),
                details={"votes": self._near_votes, "features": f},
            )

        return Event(
            kind="normal",
            confidence=1.0 - min(1.0, conf),
            timestamp_ms=now_ms(),
            details={"votes": self._fall_votes, "features": f},
        )

