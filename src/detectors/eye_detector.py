import cv2
import mediapipe as mp
from pathlib import Path

from ultralytics import YOLO


class EyeDetector:

    def __init__(self):

        # ======================================
        # PROJECT PATH
        # ======================================

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

        MODEL_PATH = (
            PROJECT_ROOT
            / "models"
            / "trained"
            / "eye_model_v1.pt"
        )

        # ======================================
        # LOAD YOLO MODEL
        # ======================================

        print("Loading eye detection model...")

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
        # EYE LANDMARKS
        # ======================================

        self.LEFT_EYE = [
            33, 133, 160, 159,
            158, 157, 173, 246,
            161, 163, 144, 145,
            153, 154, 155
        ]

        self.RIGHT_EYE = [
            362, 263, 387, 386,
            385, 384, 398, 466,
            388, 390, 373, 374,
            380, 381, 382
        ]


    # ==========================================
    # CROP REGION FROM LANDMARKS
    # ==========================================

    def crop_region(
        self,
        frame,
        landmarks,
        indices,
        padding=15
    ):

        height, width = frame.shape[:2]

        points = []

        for index in indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        x1 = max(min(x_values) - padding, 0)
        y1 = max(min(y_values) - padding, 0)

        x2 = min(max(x_values) + padding, width)
        y2 = min(max(y_values) + padding, height)

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


    # ==========================================
    # PREDICT SINGLE EYE
    # ==========================================

    def predict_eye(self, eye_crop):

        results = self.model.predict(
            eye_crop,
            imgsz=224,
            device="mps",
            verbose=False
        )

        result = results[0]

        class_id = result.probs.top1

        class_name = (
            result.names[class_id]
            .lower()
        )

        confidence = (
            result.probs.top1conf.item()
            * 100
        )

        return class_name, confidence


    # ==========================================
    # DETECT EYES
    # ==========================================

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        output = {
            "left_eye": None,
            "right_eye": None,
            "state": "unknown"
        }

        if not results.multi_face_landmarks:
            return output


        landmarks = (
            results
            .multi_face_landmarks[0]
            .landmark
        )


        # ======================================
        # CROP EYES
        # ======================================

        left_eye_crop, left_box = self.crop_region(
            frame,
            landmarks,
            self.LEFT_EYE
        )

        right_eye_crop, right_box = self.crop_region(
            frame,
            landmarks,
            self.RIGHT_EYE
        )


        # ======================================
        # PREDICT
        # ======================================

        if (
            left_eye_crop.size > 0
            and right_eye_crop.size > 0
        ):

            left_class, left_confidence = (
                self.predict_eye(
                    left_eye_crop
                )
            )

            right_class, right_confidence = (
                self.predict_eye(
                    right_eye_crop
                )
            )


            output["left_eye"] = {
                "state": left_class,
                "confidence": left_confidence,
                "box": left_box
            }

            output["right_eye"] = {
                "state": right_class,
                "confidence": right_confidence,
                "box": right_box
            }


            # ==================================
            # COMBINED STATE
            # ==================================

            if (
                left_class == "closed"
                and right_class == "closed"
            ):
                output["state"] = "closed"

            else:
                output["state"] = "open"


        return output


    # ==========================================
    # CLEANUP
    # ==========================================

    def close(self):

        self.face_mesh.close()