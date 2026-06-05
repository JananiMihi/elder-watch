from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import cv2


if __name__ == "__main__" and (not __package__):
    # Allow running as a script from the repo root: `python edge\\main.py`
    # In that case Python puts the `edge/` directory on sys.path, but not the repo root.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    __package__ = "edge"

try:
    from .alerting import AlertConfig, send_alert
    from .decision import DecisionConfig, TemporalDecision
    from .detector import StubDetector, YoloConfig, YoloTFLiteDetector
    from .features import MotionState, extract_features
    from .pose import MediaPipePoseEstimator, PoseConfig
    from .utils import load_config
except ImportError:  # When run as a script: `python edge\\main.py`
    from edge.alerting import AlertConfig, send_alert
    from edge.decision import DecisionConfig, TemporalDecision
    from edge.detector import YoloConfig, YoloTFLiteDetector
    from edge.features import MotionState, extract_features
    from edge.pose import MediaPipePoseEstimator, PoseConfig
    from edge.utils import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elder Watch edge runtime")
    p.add_argument("--config", default=None, help="Path to JSON config (optional)")
    p.add_argument("--demo", action="store_true", help="Run with stubs; no model files needed")
    p.add_argument("--camera", type=int, default=None, help="Override camera index")
    p.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    camera_index = int(args.camera if args.camera is not None else cfg.get("camera_index", 0))

    yolo_cfg = YoloConfig(**(cfg.get("yolo", {}) if isinstance(cfg.get("yolo", {}), dict) else {}))
    pose_cfg = PoseConfig(**(cfg.get("pose", {}) if isinstance(cfg.get("pose", {}), dict) else {}))
    dec_cfg = DecisionConfig(**(cfg.get("decision", {}) if isinstance(cfg.get("decision", {}), dict) else {}))
    alerts_cfg = AlertConfig(**(cfg.get("alerts", {}) if isinstance(cfg.get("alerts", {}), dict) else {}))

    detector = StubDetector() if args.demo else YoloTFLiteDetector(yolo_cfg)
    pose = MediaPipePoseEstimator(pose_cfg)
    decision = TemporalDecision(dec_cfg)
    motion = MotionState()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    print("[ElderWatch] Started")
    print(f"[ElderWatch] demo={args.demo} camera_index={camera_index}")
    print(f"[ElderWatch] yolo={asdict(yolo_cfg)}")
    print(f"[ElderWatch] pose={asdict(pose_cfg)}")
    print(f"[ElderWatch] decision={asdict(dec_cfg)}")
    print(f"[ElderWatch] alerts={asdict(alerts_cfg)}")

    frame_i = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt_s = 1.0 / float(fps) if fps and fps > 1e-6 else 1.0 / 30.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            detections = detector.detect(frame)
            # For Step 1 we only forward the max fall confidence if any.
            yolo_fall_conf = None
            for d in detections:
                if d.cls == "fall":
                    yolo_fall_conf = max(yolo_fall_conf or 0.0, d.conf)

            pose_lm = pose.estimate(frame)
            features = extract_features(pose_lm, motion, dt_s=dt_s) if pose_lm is not None else {}
            event = decision.update(yolo_fall_conf=yolo_fall_conf, pose=pose_lm, features=features)

            if event.kind != "normal":
                send_alert(event, alerts_cfg, extra=None)

            if not args.no_window:
                if features:
                    cv2.putText(
                        frame,
                        f"angle={features.get('torso_angle_from_vertical_deg', 0.0):.1f} "
                        f"com_y={features.get('com_y', 0.0):.2f} v={features.get('com_speed', 0.0):.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"event={event.kind} conf={event.confidence:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if event.kind == "normal" else (0, 0, 255),
                    2,
                )
                cv2.imshow("Elder Watch", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames and frame_i >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("[ElderWatch] Interrupted")

    cap.release()
    try:
        pose.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[ElderWatch] Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

