from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
)


# ==========================================
# CHECK PATHS
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not PRETRAINED_MODEL.exists():
    raise FileNotFoundError(
        f"Pretrained model not found:\n{PRETRAINED_MODEL}"
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

    project=str(OUTPUT_DIR),

    name="mouth_model_v1",

    exist_ok=True,

    pretrained=True
)


print("\nTraining complete!")
print(
    "Best model should be located at:\n"
    f"{OUTPUT_DIR / 'mouth_model_v1' / 'weights' / 'best.pt'}"
)