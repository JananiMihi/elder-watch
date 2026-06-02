## Training + export (desktop)

This folder is for:
- dataset preprocessing (frame extraction, resize, augmentation, split 70/15/15)
- YOLO fine-tuning (YOLOv8n)
- export pipeline (PyTorch → ONNX → TF → TFLite)
- INT8 quantization (PTQ first, QAT if needed)

### Planned scripts
- `prep_data.py`: build YOLO-format dataset from KFall / CAUCAFall / FDD
- `train_yolo.py`: fine-tune YOLOv8n using Ultralytics
- `export_tflite.py`: export + quantize to TFLite INT8

