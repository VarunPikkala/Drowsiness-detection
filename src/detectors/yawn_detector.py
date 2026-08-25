import cv2
import mediapipe as mp
from pathlib import Path

from ultralytics import YOLO


class YawnDetector:

    def __init__(self):

        # ======================================
        # PROJECT PATH
        # ======================================

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

        MODEL_PATH = (
            PROJECT_ROOT
            / "models"
            / "trained"
            / "yawn_model_v1.pt"
        )

        # ======================================
        # LOAD YOLO MODEL
        # ======================================

        print("Loading yawn detection model...")

        self.model = YOLO(str(MODEL_PATH))

        # ======================================
        # MEDIAPIPE FACE MESH
        # ======================================

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ======================================
        # MOUTH LANDMARKS
        # Outer + inner lips
        # ======================================

        self.MOUTH_LANDMARKS = [
            61, 185, 40, 39, 37, 0,
            267, 269, 270, 409, 291,
            146, 91, 181, 84, 17,
            314, 405, 321, 375, 291
        ]


    # ==========================================
    # CROP MOUTH REGION
    # ==========================================

    def crop_mouth(
        self,
        frame,
        landmarks,
        padding=25
    ):

        height, width = frame.shape[:2]

        points = []

        for index in self.MOUTH_LANDMARKS:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]

        x1 = max(min(x_values) - padding, 0)
        y1 = max(min(y_values) - padding, 0)

        x2 = min(max(x_values) + padding, width)
        y2 = min(max(y_values) + padding, height)

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


    # ==========================================
    # DETECT YAWN
    # ==========================================

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb_frame)

        output = {
            "state": "unknown",
            "confidence": 0,
            "box": None
        }

        if not results.multi_face_landmarks:
            return output

        landmarks = (
            results
            .multi_face_landmarks[0]
            .landmark
        )

        mouth_crop, mouth_box = self.crop_mouth(
            frame,
            landmarks
        )

        if mouth_crop.size == 0:
            return output

        # ======================================
        # YOLO PREDICTION
        # ======================================

        results = self.model.predict(
            mouth_crop,
            imgsz=224,
            device="mps",
            verbose=False
        )

        result = results[0]

        class_id = result.probs.top1

        class_name = result.names[class_id].lower()

        confidence = (
            result.probs.top1conf.item() * 100
        )

        output = {
            "state": class_name,
            "confidence": confidence,
            "box": mouth_box
        }

        return output


    # ==========================================
    # CLEANUP
    # ==========================================

    def close(self):

        self.face_mesh.close()