from pathlib import Path
import cv2

from inference.torchscript_classifier import TorchScriptClassifier


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# TEST EYE MODEL
# ==========================================

eye_model = TorchScriptClassifier(
    model_path=(
        PROJECT_ROOT
        / "models"
        / "deployment"
        / "eye_model_v1.torchscript"
    ),
    class_names={
        0: "closed",
        1: "open",
    },
)


# ==========================================
# TEST YAWN MODEL
# ==========================================

yawn_model = TorchScriptClassifier(
    model_path=(
        PROJECT_ROOT
        / "models"
        / "deployment"
        / "yawn_model_v1.torchscript"
    ),
    class_names={
        0: "no_yawn",
        1: "yawn",
    },
)


# ==========================================
# TEST IMAGE PATHS
# ==========================================

EYE_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "binary_datasets"
    / "eyes"
    / "test"
    / "open"
)

YAWN_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "binary_datasets"
    / "mouth"
    / "test"
    / "yawn"
)


# ==========================================
# FIND TEST IMAGES
# ==========================================

eye_images = sorted(
    path for path in EYE_TEST_DIR.iterdir()
    if path.is_file()
) if EYE_TEST_DIR.is_dir() else []

yawn_images = sorted(
    path for path in YAWN_TEST_DIR.iterdir()
    if path.is_file()
) if YAWN_TEST_DIR.is_dir() else []


if not eye_images:
    raise FileNotFoundError(
        f"No eye test images found in: {EYE_TEST_DIR}"
    )

if not yawn_images:
    raise FileNotFoundError(
        f"No yawn test images found in: {YAWN_TEST_DIR}"
    )


eye_image = eye_images[0]
yawn_image = yawn_images[0]


# ==========================================
# EYE MODEL TEST
# ==========================================

print("\n" + "=" * 50)
print("EYE MODEL TEST")
print("=" * 50)

image = cv2.imread(str(eye_image))

if image is None:
    raise ValueError(f"Could not load eye image: {eye_image}")

result = eye_model.predict(image)

print("Image:", eye_image.name)
print("Prediction:", result["class"])
print("Confidence:", f"{result['confidence'] * 100:.2f}%")
print("Probabilities:", result["probabilities"])


# ==========================================
# YAWN MODEL TEST
# ==========================================

print("\n" + "=" * 50)
print("YAWN MODEL TEST")
print("=" * 50)

image = cv2.imread(str(yawn_image))

if image is None:
    raise ValueError(f"Could not load yawn image: {yawn_image}")

result = yawn_model.predict(image)

print("Image:", yawn_image.name)
print("Prediction:", result["class"])
print("Confidence:", f"{result['confidence'] * 100:.2f}%")
print("Probabilities:", result["probabilities"])


# ==========================================
# COMPLETE
# ==========================================

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
