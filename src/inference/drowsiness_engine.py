from pathlib import Path

from inference.torchscript_classifier import TorchScriptClassifier
from logic.drowsiness_logic import DrowsinessLogic
from vision.face_roi import FaceROI


class DrowsinessEngine:

    def __init__(self):

        # ==========================================
        # PROJECT PATH
        # ==========================================

        project_root = Path(__file__).resolve().parents[2]

        # ==========================================
        # FACE / ROI DETECTOR
        # ==========================================

        print("Loading face ROI system...")

        self.face_roi = FaceROI()

        # ==========================================
        # EYE CLASSIFIER
        # ==========================================

        self.eye_classifier = TorchScriptClassifier(
            model_path=(
                project_root
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
        # YAWN CLASSIFIER
        # ==========================================

        self.yawn_classifier = TorchScriptClassifier(
            model_path=(
                project_root
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
        # DROWSINESS LOGIC
        # ==========================================

        print("Loading drowsiness decision system...")

        self.logic = DrowsinessLogic()

        print("Drowsiness engine ready.")

    # ==========================================
    # PROCESS FRAME
    # ==========================================

    def process(self, frame):

        # ------------------------------------------
        # Extract face regions
        # ------------------------------------------

        rois = self.face_roi.extract(frame)

        output = {
            "face_detected": False,

            "left_eye": None,
            "right_eye": None,

            "eye_state": "unknown",

            "mouth": None,
            "yawn_state": "unknown",

            "decision": None
        }

        # ------------------------------------------
        # Check face
        # ------------------------------------------
        if (
            rois["left_eye"] is None
            or rois["right_eye"] is None
            or rois["mouth"] is None
        ):
            self.logic.reset_active_timers()
            return output


        output["face_detected"] = True

        # ==========================================
        # EYE CLASSIFICATION
        # ==========================================

        left_eye_crop = rois["left_eye"]["crop"]
        right_eye_crop = rois["right_eye"]["crop"]

        left_result = self.eye_classifier.predict(
            left_eye_crop
        )

        right_result = self.eye_classifier.predict(
            right_eye_crop
        )

        output["left_eye"] = {
            "state": left_result["class"],
            "confidence": left_result["confidence"],
            "box": rois["left_eye"]["box"]
        }

        output["right_eye"] = {
            "state": right_result["class"],
            "confidence": right_result["confidence"],
            "box": rois["right_eye"]["box"]
        }

        # Both eyes must be closed
        # for the combined eye state to be closed.

        if (
            left_result["class"] == "closed"
            and right_result["class"] == "closed"
        ):
            eye_state = "closed"
        else:
            eye_state = "open"

        output["eye_state"] = eye_state

        # ==========================================
        # YAWN CLASSIFICATION
        # ==========================================

        mouth_crop = rois["mouth"]["crop"]

        yawn_result = self.yawn_classifier.predict(
            mouth_crop
        )

        output["mouth"] = {
            "state": yawn_result["class"],
            "confidence": yawn_result["confidence"],
            "box": rois["mouth"]["box"]
        }

        output["yawn_state"] = yawn_result["class"]

        # ==========================================
        # DROWSINESS LOGIC
        # ==========================================

        decision = self.logic.update(
            eye_state,
            yawn_result["class"]
        )

        output["decision"] = decision

        return output

    # ==========================================
    # CLEANUP
    # ==========================================

    def close(self):

        self.face_roi.close()