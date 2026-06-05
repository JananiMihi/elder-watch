from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO weights to TFLite (optionally INT8)")
    p.add_argument("--weights", required=True, help="Trained weights path (e.g., best.pt)")
    p.add_argument("--out", required=True, help="Output .tflite path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 export (requires a dataset for calibration; pass --data)",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml (recommended for INT8 calibration; also used by exporter)",
    )
    p.add_argument("--device", default=None, help="Device string like 0, cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ultralytics is required. Install it with: pip install ultralytics\n" f"Original error: {e}"
        )

    model = YOLO(str(weights))

    export_kwargs = {
        "format": "tflite",
        "imgsz": int(args.imgsz),
        "device": args.device,
    }
    if args.data:
        export_kwargs["data"] = str(Path(args.data))
    if args.int8:
        export_kwargs["int8"] = True

    exported = model.export(**export_kwargs)
    exported_path = Path(str(exported)) if exported is not None else None

    # Ultralytics export returns a path, but in some versions it returns None.
    if exported_path is None or not exported_path.exists():
        # Try conventional export locations.
        candidates = list(weights.parent.rglob("*.tflite"))
        if not candidates:
            raise SystemExit("Export did not produce a .tflite file. Ensure TensorFlow is installed.")
        exported_path = max(candidates, key=lambda p: p.stat().st_mtime)

    shutil.copy2(exported_path, out)
    print(f"[export] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

