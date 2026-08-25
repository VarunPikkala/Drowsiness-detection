from ultralytics import YOLO
from pathlib import Path
import shutil
import os


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "processed"
RUNS_PATH = PROJECT_ROOT / "runs"
MODELS_PATH = PROJECT_ROOT / "models"

RUNS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)


# ==========================================
# PRETRAINED MODEL
# ==========================================

PRETRAINED_MODEL = MODELS_PATH / "yolo11n-cls.pt"

# Download model into project/models if missing
if not PRETRAINED_MODEL.exists():

    print("Pretrained model not found.")
    print("Downloading YOLO11 classification model...\n")

    # Temporarily switch to models directory
    os.chdir(MODELS_PATH)

    model = YOLO("yolo11n-cls.pt")

    # Return to project root
    os.chdir(PROJECT_ROOT)

else:
    print(f"Loading pretrained model from:\n{PRETRAINED_MODEL}\n")

    model = YOLO(str(PRETRAINED_MODEL))


# ==========================================
# TRAIN
# ==========================================

print("Starting training...\n")

results = model.train(
    data=str(DATASET_PATH),
    epochs=30,
    imgsz=224,
    batch=32,
    device="mps",
    project=str(RUNS_PATH),
    name="yolo_features_v1",
    exist_ok=True
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

best_model = Path(results.save_dir) / "weights" / "best.pt"

final_model_path = MODELS_PATH / "drowsiness_features_v1.pt"

if best_model.exists():
    shutil.copy2(best_model, final_model_path)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nFinal model saved to:\n{final_model_path}")

else:
    print("\nWARNING: best.pt was not found.")


print("\nDone!")