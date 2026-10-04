import cv2
import torch
from pathlib import Path


class TorchScriptClassifier:

    def __init__(self, model_path, class_names):
        """
        Load a TorchScript classification model.

        Args:
            model_path: Path to the .torchscript model.
            class_names: Dictionary mapping class IDs to class names.
        """

        self.model_path = Path(model_path)
        self.class_names = class_names

        print(f"Loading TorchScript model: {self.model_path}")

        self.model = torch.jit.load(
            str(self.model_path),
            map_location="cpu"
        )

        self.model.eval()

        print("TorchScript model loaded successfully.")

    # ==========================================
    # PREPROCESS IMAGE
    # ==========================================

    def preprocess(self, image):
        """
        Convert an OpenCV BGR image into the
        tensor expected by the TorchScript model.
        """

        # OpenCV uses BGR.
        # Model expects RGB.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize to deployment resolution.
        image = cv2.resize(
            image,
            (320, 320),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert HWC → CHW
        image = image.transpose(
            2, 0, 1
        )

        # Convert to float32 and normalize 0–255 → 0–1
        image = torch.from_numpy(
            image
        ).float() / 255.0

        # Add batch dimension
        image = image.unsqueeze(0)

        return image

    # ==========================================
    # PREDICT
    # ==========================================

    def predict(self, image):
        """
        Run classification on an OpenCV image.

        Returns:
            class_name
            confidence
            probabilities
        """

        if image is None or image.size == 0:
            return {
                "class": "unknown",
                "confidence": 0.0,
                "probabilities": None
            }

        tensor = self.preprocess(image)

        with torch.no_grad():

            output = self.model(tensor)

        # Some TorchScript models can return
        # tuple/list outputs.
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Convert logits → probabilities
        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        # Highest probability class
        class_id = int(
            torch.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_id]
        )

        class_name = self.class_names.get(
            class_id,
            "unknown"
        )

        return {
            "class": class_name,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }