## Training + export (desktop)

This folder is for:
- dataset preprocessing (frame extraction, resize, augmentation, split 70/15/15)
- YOLO fine-tuning (YOLOv8n)
- export pipeline (PyTorch → ONNX → TF → TFLite)
- INT8 quantization (PTQ first, QAT if needed)

### Scripts (implemented)

1) Prepare a YOLO dataset (baseline via pseudo-labeling)

This script assumes you have two folders containing either videos or image frames:
- `fall/`: fall videos
- `adl/`: normal activities / ADL videos

It samples frames, runs a pretrained YOLO model to find the best `person` box, and writes
a YOLO-format detection dataset with two classes:
- `0: person` (ADL videos)
- `1: fall` (fall videos)

```bash
python training\prep_data.py --fall data\raw\fall --adl data\raw\adl --out data\yolo --fps 5 --imgsz 640
```

This creates `data\yolo\data.yaml`.

2) Fine-tune YOLO

```bash
python training\train_yolo.py --data data\yolo\data.yaml --model yolov8n.pt --epochs 80 --imgsz 640
```

Ultralytics will write a `best.pt` under `runs/train/...`.

3) Export to TFLite

```bash
python training\export_tflite.py --weights runs\train\elderwatch\weights\best.pt --out models\yolo_int8.tflite --imgsz 640 --int8 --data data\yolo\data.yaml
```

If INT8 export fails on your machine, try without `--int8` first.

4) Run on edge runtime

```bash
python -m edge.main
```

The edge runtime loads the model path from `edge/config.json` (see `yolo.model_path`).

