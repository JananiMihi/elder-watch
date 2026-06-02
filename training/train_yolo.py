from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO training placeholder for Elder Watch")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("TODO: Implement ultralytics YOLOv8n fine-tuning here.")
    print(f"data={args.data} epochs={args.epochs} imgsz={args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

