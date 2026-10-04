from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "models" / "deployment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# MODELS
# ==========================================

MODELS = {
    "eye": MODEL_DIR / "eye_model_v1.pt",
    "yawn": MODEL_DIR / "yawn_model_v1.pt",
}


# ==========================================
# EXPORT
# ==========================================

for name, model_path in MODELS.items():

    print("\n" + "=" * 50)
    print(f"Exporting {name} model")
    print("=" * 50)

    print(f"Input : {model_path}")

    model = YOLO(str(model_path))

    print(f"Classes: {model.names}")

    exported_path = model.export(
    format="torchscript",
    imgsz=320,
    )

    print(f"Ultralytics export: {exported_path}")

    # Ultralytics normally creates the exported file beside
    # the original .pt file. Move/copy it into deployment/.
    exported_path = Path(exported_path)

    destination = OUTPUT_DIR / exported_path.name

    exported_path.replace(destination)

    print(f"Saved deployment model: {destination}")


print("\n" + "=" * 50)
print("EXPORT COMPLETE")
print("=" * 50)
