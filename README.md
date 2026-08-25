# Driver Drowsiness Detection using YOLO

A real-time **Driver Drowsiness Detection System** built using deep learning and computer vision.

This project uses a **YOLO11 classification model** to detect important visual indicators of driver fatigue and drowsiness:

- Open Eyes
- Closed Eyes
- Yawning
- No Yawning

These visual features are then analyzed over time to determine the driver's overall state:

- 🟢 Awake
- 🟡 Fatigued
- 🔴 Drowsy

The project is designed as an edge AI prototype and can later be optimized for deployment on devices such as the **Raspberry Pi 5**.

---

## 🚀 Project Architecture

```text
                    ┌──────────────┐
                    │    Camera    │
                    └──────┬───────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   YOLO11 Classifier  │
                └──────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Open Eyes        Closed Eyes        Yawning
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Temporal Analysis │
                 │   & State Logic   │
                 └─────────┬─────────┘
                           ▼
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           AWAKE       FATIGUED      DROWSY