## Elder Watch (CV-based Elder Monitoring)

This project implements the **Elder Watch** system described in `Mini_Project_Proposal_41 (1).pdf`:

- **On-device** (privacy-preserving) monitoring using a ceiling/overhead camera
- **Fall / near-fall / immobility** risk detection using:
  - person detection (YOLO exported to **TFLite**)
  - pose estimation (**MediaPipe Pose**)
  - motion features + temporal smoothing
- **Guardian alerts** (Twilio SMS or alternative)

### Repo layout
- `edge/`: Raspberry Pi runtime (camera → detection → pose → decision → alert)
- `training/`: dataset prep, YOLO training, export + quantization scripts

### Quick start (dev PC)
1. Create venv and install:

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

2. Run edge app in **demo mode** (no model files needed yet):

```bash
python edge\main.py --demo
```

### Next steps (implementation order)
1. Get the edge pipeline running end-to-end (even with stubs).
2. Train/fine-tune YOLO on KFall / CAUCAFall / FDD.
3. Export → quantize to INT8 TFLite.
4. Replace stubs with real TFLite inference on Raspberry Pi.

