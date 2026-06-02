from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset prep placeholder for Elder Watch")
    p.add_argument("--out", required=True, help="Output dataset directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(
        "TODO: Implement frame extraction (30 FPS), resize (640), augmentation, and YOLO labels.\n",
        encoding="utf-8",
    )
    print(f"Wrote placeholder to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

