from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    fall_conf: float = 0.6
    # Expected class names in the trained model.
    person_class: str = "person"
    fall_class: str = "fall"


class BaseDetector:
    def warmup(self) -> None:
        return

    def detect(self, frame_bgr) -> list[Detection]:
        raise NotImplementedError


class StubDetector(BaseDetector):
    """Demo stub: returns no detections so the pipeline can run without models."""

    def detect(self, frame_bgr) -> list[Detection]:
        return []


class YoloTFLiteDetector:
    """Runs detection using Ultralytics.

    Despite the name, this supports any model path Ultralytics supports
    (e.g. .pt, .onnx, .tflite). On Raspberry Pi you can still point this
    at a TFLite-exported model.
    """

    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        self._model: Any | None = None

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            # Keep behavior friendly: allow running even before a model is trained.
            self._model = None
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            # Ultralytics is optional in requirements.txt; if missing, detector becomes a stub.
            self._model = None
            return

        self._model = YOLO(str(model_path))

    def warmup(self) -> None:
        # Ultralytics lazily initializes; a single predict warms up.
        if self._model is None:
            return
        try:
            self._model.predict(source=None, verbose=False)
        except Exception:
            # Some backends don't accept None source; ignore warmup.
            return

    def detect(self, frame_bgr) -> list[Detection]:
        if self._model is None or frame_bgr is None:
            return []

        # Ultralytics expects RGB for numpy arrays sometimes; it also accepts BGR.
        # We'll pass BGR frame directly and rely on Ultralytics preprocessing.
        res = self._model.predict(
            source=frame_bgr,
            imgsz=int(self.cfg.input_size),
            conf=min(float(self.cfg.person_conf), float(self.cfg.fall_conf), 0.95),
            verbose=False,
        )
        if not res:
            return []

        r0 = res[0]
        names = getattr(r0, "names", {}) or {}
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        out: list[Detection] = []
        xyxy = boxes.xyxy
        cls = boxes.cls
        conf = boxes.conf

        # Convert tensors to CPU lists when needed.
        try:
            xyxy = xyxy.cpu().numpy()
            cls = cls.cpu().numpy()
            conf = conf.cpu().numpy()
        except Exception:
            pass

        for (x1, y1, x2, y2), cidx, cconf in zip(xyxy, cls, conf):
            try:
                class_name = names.get(int(cidx), str(int(cidx)))
            except Exception:
                class_name = str(cidx)

            score = float(cconf)
            if class_name == self.cfg.person_class and score < float(self.cfg.person_conf):
                continue
            if class_name == self.cfg.fall_class and score < float(self.cfg.fall_conf):
                continue
            if class_name not in (self.cfg.person_class, self.cfg.fall_class):
                continue

            out.append(
                Detection(
                    cls=class_name,
                    conf=score,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )

        return out

