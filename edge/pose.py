from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python import vision as mp_vision


@dataclass(frozen=True)
class PoseLandmarks:
    # normalized [0..1] landmark coordinates in image space
    # store as list of (x,y,visibility) tuples
    landmarks: list[tuple[float, float, float]]


@dataclass
class PoseConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str = "models/pose_landmarker_lite.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )


class MediaPipePoseEstimator:
    def __init__(self, cfg: PoseConfig):
        self.cfg = cfg
        # MediaPipe >=0.10 exposes the Tasks API (mp.tasks) instead of mp.solutions.
        # We use IMAGE mode for simplicity (per-frame processing).
        model_path = self._ensure_model(cfg.model_path, cfg.model_url)
        base = mp_python_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=float(cfg.min_detection_confidence),
            min_pose_presence_confidence=float(cfg.min_detection_confidence),
            min_tracking_confidence=float(cfg.min_tracking_confidence),
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model(model_path: str, model_url: str) -> str:
        from pathlib import Path

        p = Path(model_path)
        if p.exists():
            return str(p)
        p.parent.mkdir(parents=True, exist_ok=True)

        import requests

        r = requests.get(model_url, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
        return str(p)

    def estimate(self, frame_bgr) -> PoseLandmarks | None:
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self._landmarker.detect(image)
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks[0]
        # Tasks API gives normalized landmark coordinates (x,y in [0,1]) and visibility.
        pts: list[tuple[float, float, float]] = [
            (float(p.x), float(p.y), float(getattr(p, "visibility", 1.0))) for p in lm
        ]
        return PoseLandmarks(landmarks=pts)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

