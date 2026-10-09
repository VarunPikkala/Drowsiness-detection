
from pathlib import Path
import shutil

from ultralytics import YOLO


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "binary_datasets"
    / "mouth"
)

PRETRAINED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "pretrained"
    / "yolo11n-cls.pt"
)

RUNS_OUTPUT = PROJECT_ROOT / "runs"

MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "trained"
)


# ==========================================
# CHECK PATHS
# ==========================================

if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not PRETRAINED_MODEL.is_file():
    raise FileNotFoundError(
        f"Pretrained model not found:\n{PRETRAINED_MODEL}"
    )

MODEL_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading pretrained YOLO11 classification model...")

model = YOLO(str(PRETRAINED_MODEL))


# ==========================================
# TRAIN
# ==========================================

print("\nStarting mouth/yawn model training...\n")

model.train(
    data=str(DATASET_PATH),
    epochs=15,
    imgsz=224,
    batch=32,
    device="mps",
    workers=0,
    project=str(RUNS_OUTPUT),
    name="mouth_model_v1",
    exist_ok=True,
    pretrained=True,
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

best_model_path = (
    RUNS_OUTPUT
    / "mouth_model_v1"
    / "weights"
    / "best.pt"
)

target_model_path = (
    MODEL_OUTPUT
    / "yawn_model_v1.pt"
)

if not best_model_path.is_file():
    raise FileNotFoundError(
        "Training finished, but best.pt was not found:\n"
        f"{best_model_path}"
    )

shutil.copy2(
    best_model_path,
    target_model_path
)

print("\n" + "=" * 50)
print("MOUTH MODEL TRAINING COMPLETE")
print("=" * 50)

print(f"Training checkpoint:\n{best_model_path}")
print(f"\nTrained model saved to:\n{target_model_path}")
