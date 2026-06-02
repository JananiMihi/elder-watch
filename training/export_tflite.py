from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export + quantize placeholder (ONNX→TF→TFLite)")
    p.add_argument("--weights", required=True, help="Trained weights path (e.g., best.pt)")
    p.add_argument("--out", required=True, help="Output .tflite path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("TODO: Implement export pipeline: PyTorch → ONNX → TF → TFLite + INT8 PTQ.")
    print(f"weights={args.weights} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

