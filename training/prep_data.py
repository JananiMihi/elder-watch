from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def _write_yolo_label(path: Path, *, class_id: int, xyxy: tuple[int, int, int, int], w: int, h: int) -> None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    nw = bw / w
    nh = bh / h

    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")


def _iter_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _iter_media(folder: Path) -> list[tuple[str, Path]]:
    """Return a list of (kind, path) where kind is 'video' or 'image'."""
    out: list[tuple[str, Path]] = []
    out.extend([("video", p) for p in _iter_videos(folder)])
    out.extend([("image", p) for p in _iter_images(folder)])
    # Keep deterministic ordering.
    out.sort(key=lambda x: str(x[1]).lower())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a YOLO detection dataset (pseudo-labeling)")
    p.add_argument("--fall", required=True, help="Folder with fall videos")
    p.add_argument("--adl", required=True, help="Folder with ADL / normal videos")
    p.add_argument("--out", required=True, help="Output dataset directory")
    p.add_argument("--imgsz", type=int, default=640, help="Image size to save (square)")
    p.add_argument("--fps", type=float, default=5.0, help="Frame sampling FPS (kept low to reduce duplicates)")
    p.add_argument("--max-frames-per-video", type=int, default=300, help="Cap extracted frames per video")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="0.7,0.15,0.15", help="train,val,test split")
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Pretrained YOLO weights used to pseudo-label person boxes",
    )
    p.add_argument("--det-conf", type=float, default=0.4, help="Confidence for pseudo-label person detection")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(int(args.seed))

    fall_dir = Path(args.fall)
    adl_dir = Path(args.adl)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        split_train, split_val, split_test = [float(x) for x in str(args.split).split(",")]
    except Exception:
        raise SystemExit("--split must be like 0.7,0.15,0.15")
    if abs((split_train + split_val + split_test) - 1.0) > 1e-6:
        raise SystemExit("--split values must sum to 1.0")

    # Prepare output folders.
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Load pseudo-label model.
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ultralytics is required for pseudo-labeling. Install it with: pip install ultralytics\n"
            f"Original error: {e}"
        )

    pseudo = YOLO(str(args.weights))

    fall_media = _iter_media(fall_dir)
    adl_media = _iter_media(adl_dir)
    if not fall_media or not adl_media:
        raise SystemExit("No media found (videos or images). Check --fall and --adl paths.")

    # Build a flat list of (kind, path, class_id) where class_id matches YOLO data.yaml.
    # 0=person (ADL), 1=fall
    items: list[tuple[str, Path, int]] = [(k, p, 1) for k, p in fall_media] + [(k, p, 0) for k, p in adl_media]
    random.shuffle(items)

    n = len(items)
    n_train = int(n * split_train)
    n_val = int(n * split_val)

    def split_of(i: int) -> str:
        if i < n_train:
            return "train"
        if i < n_train + n_val:
            return "val"
        return "test"

    total_frames = 0
    kept_frames = 0

    for i, (kind, media_path, class_id) in enumerate(items):
        split = split_of(i)
        imgsz = int(args.imgsz)

        def process_frame(frame_bgr, *, frame_id: str) -> bool:
            nonlocal kept_frames
            if frame_bgr is None:
                return False
            frame_r = cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            pred = pseudo.predict(source=frame_r, conf=float(args.det_conf), imgsz=imgsz, verbose=False)
            if not pred:
                return False
            r0 = pred[0]
            boxes = getattr(r0, "boxes", None)
            names = getattr(r0, "names", {}) or {}
            if boxes is None or boxes.xyxy is None:
                return False

            xyxy = boxes.xyxy
            cls = boxes.cls
            conf = boxes.conf
            try:
                xyxy = xyxy.cpu().numpy()
                cls = cls.cpu().numpy()
                conf = conf.cpu().numpy()
            except Exception:
                pass

            best = None
            for (x1, y1, x2, y2), cidx, cconf in zip(xyxy, cls, conf):
                name = names.get(int(cidx), str(int(cidx)))
                if name != "person":
                    continue
                score = float(cconf)
                if best is None or score > best[0]:
                    best = (score, (int(x1), int(y1), int(x2), int(y2)))
            if best is None:
                return False

            stem = f"{media_path.stem}_{frame_id}"
            img_path = out / "images" / split / f"{stem}.jpg"
            lbl_path = out / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(img_path), frame_r)
            _write_yolo_label(lbl_path, class_id=class_id, xyxy=best[1], w=imgsz, h=imgsz)
            kept_frames += 1
            return True

        if kind == "image":
            frame = cv2.imread(str(media_path))
            total_frames += 1
            ok = process_frame(frame, frame_id="img")
            print(f"[prep] {split}: {media_path.name} -> {1 if ok else 0} frames")
            continue

        # kind == 'video'
        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            print(f"[prep] WARN: could not open {media_path}")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        native_fps = float(native_fps) if native_fps and native_fps > 1e-6 else 30.0
        step = max(1, int(round(native_fps / float(args.fps))))

        frame_idx = 0
        saved_for_video = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            frame_idx += 1
            if (frame_idx % step) != 0:
                continue
            if args.max_frames_per_video and saved_for_video >= int(args.max_frames_per_video):
                break

            saved = process_frame(frame, frame_id=f"{frame_idx:06d}")
            if saved:
                saved_for_video += 1

        cap.release()
        print(f"[prep] {split}: {media_path.name} -> {saved_for_video} frames")

    # Write data.yaml.
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: person",
                "  1: fall",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (out / "README.txt").write_text(
        "Elder Watch YOLO dataset generated by pseudo-labeling person boxes.\n"
        "- classes: 0=person (ADL), 1=fall (fall videos)\n"
        "- NOTE: This is a baseline dataset. For best accuracy, manually correct labels/boxes.\n",
        encoding="utf-8",
    )

    print(f"[prep] Done. total_frames={total_frames} kept_frames={kept_frames}")
    print(f"[prep] data.yaml -> {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

