from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for Elder Watch (person vs fall)")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model/weights (e.g. yolov8n.pt, yolo11n.pt, or a previous best.pt)",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="Device string like 0, cpu")
    p.add_argument("--project", default="runs/train", help="Ultralytics project output folder")
    p.add_argument("--name", default="elderwatch", help="Ultralytics run name")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ultralytics is required. Install it with: pip install ultralytics\n" f"Original error: {e}"
        )

    model = YOLO(str(args.model))
    model.train(
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        resume=bool(args.resume),
        # Helpful defaults for a 2-class detector (person vs fall)
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        patience=25,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Print validation metrics for convenience.
    metrics = model.val(data=str(data), imgsz=int(args.imgsz), device=args.device)
    try:
        print("[train] metrics:", metrics.results_dict)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

