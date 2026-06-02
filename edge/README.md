## Edge runtime (Raspberry Pi)

This folder contains the on-device pipeline:

1) Capture frame (camera/video)  
2) Detect person (YOLO TFLite)  
3) Pose landmarks (MediaPipe Pose)  
4) Feature extraction + temporal smoothing  
5) Decision (fall / near-fall / immobile / normal)  
6) Alert (Twilio / webhook / etc.)

### Run demo mode (no models required)

```bash
python edge\main.py --demo
```

