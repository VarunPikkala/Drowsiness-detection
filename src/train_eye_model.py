from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "binary_datasets"
    / "eyes"
)

PRETRAINED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "pretrained"
    / "yolo11n-cls.pt"
)

MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "trained"
)

RUNS_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "eye_model_v1"
)


# ==========================================
# CHECKS
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not PRETRAINED_MODEL.exists():
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

print("Loading YOLO11 classification model...")

model = YOLO(str(PRETRAINED_MODEL))


# ==========================================
# TRAIN
# ==========================================

print("\nStarting eye model training...\n")

results = model.train(

    data=str(DATASET_PATH),

    epochs=15,

    imgsz=224,

    batch=32,

    device="mps",

    project=str(PROJECT_ROOT / "runs"),

    name="eye_model_v1",

    exist_ok=True,

    pretrained=True
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

best_model_path = (
    PROJECT_ROOT
    / "runs"
    / "eye_model_v1"
    / "weights"
    / "best.pt"
)

target_model_path = (
    MODEL_OUTPUT
    / "eye_model_v1.pt"
)


if best_model_path.exists():

    import shutil

    shutil.copy2(
        best_model_path,
        target_model_path
    )

    print("\nTraining complete!")

    print(
        f"\nBest model saved to:\n"
        f"{target_model_path}"
    )

else:

    print(
        "\nTraining finished, "
        "but best.pt was not found."
    )