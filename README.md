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
\.\.venv\Scripts\python -m pip install -r requirements.txt
```

> Note (Windows PowerShell): to activate the venv, use `\.\.venv\Scripts\Activate.ps1`.
> The `activate` script without `.ps1` is for CMD and won't switch `python` in PowerShell.

2. Run edge app in **demo mode** (no model files needed yet):

```bash
python -m edge.main --demo
```

3. Train + export a real YOLO model (desktop):

- See [training/README.md](training/README.md)

Then run without `--demo` to enable YOLO inference.

### Next steps (implementation order)
1. Get the edge pipeline running end-to-end (even with stubs).
2. Train/fine-tune YOLO on KFall / CAUCAFall / FDD.
3. Export → quantize to INT8 TFLite.
4. Replace stubs with real TFLite inference on Raspberry Pi.

