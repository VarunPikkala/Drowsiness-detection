# Driver Drowsiness Detection using YOLO11

A real-time **Driver Drowsiness Detection System** built using deep learning and computer vision, designed as an edge AI prototype for driver fatigue monitoring.

The system uses two YOLO11 classification models to analyze eye state and yawning behavior. These visual indicators are combined with temporal logic to estimate the driver's state.

## Features

- **Eye classification:** Open and closed eyes.
- **Yawn classification:** Yawning and no yawning.
- **Face-region extraction:** MediaPipe-based detection of eye and mouth regions.
- **TorchScript inference:** Deployment-oriented model artifacts for inference.
- **Temporal analysis:** Uses eye-closure duration and confirmed yawns to estimate fatigue.
- **Live visualization:** Displays detected regions, predictions, confidence scores, and status.
- **Automated tests:** Tests for drowsiness logic and engine control flow.
- **Edge AI direction:** Designed for future deployment on Raspberry Pi hardware.

## Project Architecture

```text
Camera
  |
  v
OpenCV Frame Capture
  |
  v
MediaPipe Face ROI Extraction
  |
  +-------------------+-------------------+
  |                   |                   |
  v                   v                   v
Left Eye ROI      Right Eye ROI       Mouth ROI
  |                   |                   |
  +---------+---------+                   |
            |                             |
            v                             v
      YOLO11 Eye Classifier        YOLO11 Yawn Classifier
            |                             |
            v                             v
       Eye State                    Yawn State
            |                             |
            +-------------+---------------+
                          |
                          v
                 Temporal Drowsiness Logic
                          |
                          v
                 AWAKE / FATIGUED / DROWSY
```

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLO11 classification
- TorchScript
- OpenCV
- MediaPipe
- NumPy
- pytest

## Repository Structure

```text
Drowsiness_detection_project/
├── data/
│   └── binary_datasets/
│       ├── eyes/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── mouth/
│           ├── train/
│           ├── val/
│           └── test/
├── models/
│   ├── pretrained/
│   │   └── yolo11n-cls.pt
│   ├── trained/
│   │   ├── eye_model_v1.pt
│   │   └── yawn_model_v1.pt
│   └── deployment/
│       ├── eye_model_v1.torchscript
│       └── yawn_model_v1.torchscript
├── src/
│   ├── inference/
│   │   ├── drowsiness_engine.py
│   │   ├── torchscript_classifier.py
│   │   ├── test_drowsiness_engine.py
│   │   └── test_torchscript_classifier.py
│   ├── logic/
│   │   └── drowsiness_logic.py
│   ├── optimization/
│   │   └── export_torchscript.py
│   ├── vision/
│   │   ├── face_roi.py
│   │   └── test_face_roi.py
│   └── main.py
├── training/
│   ├── prepare_binary_datasets.py
│   ├── train_eye_model.py
│   └── train_mouth_model.py
├── tests/
│   ├── test_drowsiness_logic.py
│   └── test_drowsiness_engine.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 is the version used in the current development environment.
- A webcam for live inference.
- The required model files and dataset directories.
- A compatible installation of PyTorch, Ultralytics, OpenCV, and MediaPipe.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/VarunPikkala/Drowsiness-detection.git
cd Drowsiness-detection
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the required model files are present:

```text
models/deployment/eye_model_v1.torchscript
models/deployment/yawn_model_v1.torchscript
```

**Note:** PyTorch and MediaPipe installation compatibility can depend on your operating system, Python version, and hardware architecture.

## Running the Application

From the repository root, run:

```bash
PYTHONPATH=src python src/main.py
```

Allow camera access when prompted by your operating system.

The application uses the camera to extract facial regions, classify eye and mouth states, and apply temporal logic to estimate drowsiness.

## Testing

Run the automated test suite:

```bash
PYTHONPATH=src python -m pytest -v
```

The tests currently cover:

- Initial awake status.
- Eye-closure duration thresholds.
- Eye timer reset when eyes reopen.
- Yawn confirmation and duplicate prevention.
- Yawn history expiration.
- Resetting active timers while preserving confirmed-yawn history.
- Engine behavior when facial regions are unavailable.
- Engine classification flow with mocked components.

These tests verify decision logic and selected engine behaviors. They do not establish real-world detection accuracy or validate the complete live-camera pipeline.

## Training

The project includes separate training scripts for eye and mouth classification.

Train the eye model:

```bash
PYTHONPATH=src python training/train_eye_model.py
```

Train the mouth/yawn model:

```bash
PYTHONPATH=src python training/train_mouth_model.py
```

The current training scripts use the pretrained YOLO11 classification checkpoint and configure Apple Silicon MPS for training.

Training settings, including epochs, image size, batch size, and device, are defined in the respective scripts.

**Important:** Training overwrites or updates artifacts depending on the configured output paths. Back up important checkpoints before retraining.

## TorchScript Export

The project contains an export script:

```bash
PYTHONPATH=src python src/optimization/export_torchscript.py
```

The script exports the trained classifiers for TorchScript inference. Confirm that the expected deployment artifacts exist before launching the application.

## Drowsiness Decision Logic

The current temporal decision rules are:

| Indicator | Configured threshold |
|---|---:|
| Eye closure associated with fatigue | 1.5 seconds |
| Eye closure associated with drowsiness | 3.0 seconds |
| Yawn confirmation duration | 0.8 seconds |
| Yawn history window | 60 seconds |
| Yawns associated with fatigue | 3 confirmed yawns |

The system combines these rules to produce one of three statuses:

- **AWAKE:** No fatigue threshold is currently met.
- **FATIGUED:** The configured eye-closure or repeated-yawn threshold is met.
- **DROWSY:** The configured prolonged eye-closure threshold is met.

These are prototype decision rules, not a medical assessment or a guarantee of driver safety.

## Raspberry Pi Deployment

The project is intended to be evaluated for edge deployment, including Raspberry Pi hardware.

Before deployment:

1. Verify compatibility of the operating system and Python version with the required libraries.
2. Install a compatible PyTorch and MediaPipe environment.
3. Confirm that the TorchScript models load and produce predictions on the target device.
4. Measure inference latency, memory usage, CPU utilization, and sustained temperature.
5. Test camera behavior and detection stability under realistic lighting conditions.

The Mac development environment and the Raspberry Pi deployment environment may require different dependency configurations. Performance and compatibility must be measured on the target hardware before claiming real-time operation.

## Limitations

- Predictions can be affected by lighting, camera position, glasses, occlusion, and face-detection quality.
- A classification confidence score does not guarantee a correct prediction.
- Temporal rules may produce false positives or false negatives.
- The current automated tests do not prove real-world accuracy.
- Raspberry Pi performance and compatibility have not been established by the unit tests alone.

This project is an educational and engineering prototype. It should not be treated as a substitute for established driver-safety systems.

## Future Improvements

- Add continuous integration with GitHub Actions.
- Expand automated tests for face-loss and recovery scenarios.
- Evaluate classification performance on a held-out dataset.
- Measure inference latency and resource usage on Raspberry Pi hardware.
- Improve logging, configuration, and deployment documentation.

## License

Add the appropriate license and attribution details before distributing the project.
