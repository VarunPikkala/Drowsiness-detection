from pathlib import Path

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
        1: "open"
    }
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
        1: "yawn"
    }
)


# ==========================================
# TEST IMAGES
# ==========================================

eye_image = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Test"
    / "Open_Eyes"
)

yawn_image = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Test"
    / "Yawn"
)


# Pick first image from each folder
eye_image = list(eye_image.glob("*"))[0]
yawn_image = list(yawn_image.glob("*"))[0]


# ==========================================
# LOAD OPENCV
# ==========================================

import cv2


# ==========================================
# EYE TEST
# ==========================================

print("\n" + "=" * 50)
print("EYE MODEL TEST")
print("=" * 50)

image = cv2.imread(str(eye_image))

result = eye_model.predict(image)

print("Image:", eye_image.name)
print("Prediction:", result["class"])
print(
    "Confidence:",
    f"{result['confidence'] * 100:.2f}%"
)
print("Probabilities:", result["probabilities"])


# ==========================================
# YAWN TEST
# ==========================================

print("\n" + "=" * 50)
print("YAWN MODEL TEST")
print("=" * 50)

image = cv2.imread(str(yawn_image))

result = yawn_model.predict(image)

print("Image:", yawn_image.name)
print("Prediction:", result["class"])
print(
    "Confidence:",
    f"{result['confidence'] * 100:.2f}%"
)
print("Probabilities:", result["probabilities"])


print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)