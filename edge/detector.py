from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    cls: str  # "person" | "fall" | ...
    conf: float
    # xyxy pixel coords
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class YoloConfig:
    model_path: str = "models/yolo_int8.tflite"
    input_size: int = 640
    person_conf: float = 0.7


class YoloTFLiteDetector:
    """
    Placeholder for Step 1–2.
    In Step 7 you’ll replace `detect()` with real TFLite inference.
    """

    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg

    def warmup(self) -> None:
        return

    def detect(self, frame_bgr) -> list[Detection]:
        # Demo stub: return empty list so the pipeline still runs.
        return []

