from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .pose import PoseLandmarks


def _avg_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_deg(vx: float, vy: float) -> float:
    # angle between vector and +Y axis (vertical). 0 = perfectly vertical, 90 = horizontal.
    # Use atan2(|vx|, |vy|) to make it orientation-agnostic.
    return math.degrees(math.atan2(abs(vx), max(1e-9, abs(vy))))


@dataclass
class MotionState:
    last_com: tuple[float, float] | None = None
    last_speed: float = 0.0
    immobile_frames: int = 0


def extract_features(pose: PoseLandmarks, motion: MotionState, *, dt_s: float) -> dict[str, Any]:
    """
    Features aligned with the proposal:
    - Center-of-mass height (normalized y)
    - Torso inclination (angle vs vertical)
    - Simple COM speed for sudden change / inactivity proxy
    """
    lm = pose.landmarks
    if len(lm) < 25:
        return {}

    # Indices from MediaPipe Pose:
    # 11/12 shoulders, 23/24 hips
    l_sh, r_sh = lm[11], lm[12]
    l_hip, r_hip = lm[23], lm[24]

    sh = _avg_xy(l_sh, r_sh)
    hip = _avg_xy(l_hip, r_hip)

    com = ((sh[0] + hip[0]) / 2.0, (sh[1] + hip[1]) / 2.0)
    torso_vx = sh[0] - hip[0]
    torso_vy = sh[1] - hip[1]
    torso_angle_from_vertical = _angle_deg(torso_vx, torso_vy)

    # Motion: COM speed
    speed = 0.0
    if motion.last_com is not None and dt_s > 0:
        dx = com[0] - motion.last_com[0]
        dy = com[1] - motion.last_com[1]
        speed = math.sqrt(dx * dx + dy * dy) / dt_s

    motion.last_com = com
    motion.last_speed = speed

    return {
        "com_x": com[0],
        "com_y": com[1],
        "torso_angle_from_vertical_deg": torso_angle_from_vertical,
        "com_speed": speed,
    }

