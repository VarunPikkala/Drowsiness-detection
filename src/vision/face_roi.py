import cv2
import mediapipe as mp


class FaceROI:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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

        self.MOUTH = [
            61, 185, 40, 39, 37, 0,
            267, 269, 270, 409, 291,
            146, 91, 181, 84, 17,
            314, 405, 321, 375, 291
        ]

    def _crop_region(self, frame, landmarks, indices, padding):

        height, width = frame.shape[:2]

        points = []

        for index in indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = max(min(xs) - padding, 0)
        y1 = max(min(ys) - padding, 0)

        x2 = min(max(xs) + padding, width)
        y2 = min(max(ys) + padding, height)

        crop = frame[y1:y2, x1:x2]

        return crop, (x1, y1, x2, y2)

    def extract(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        output = {
            "left_eye": None,
            "right_eye": None,
            "mouth": None
        }

        if not results.multi_face_landmarks:
            return output

        landmarks = (
            results
            .multi_face_landmarks[0]
            .landmark
        )

        left_eye, left_box = self._crop_region(
            frame,
            landmarks,
            self.LEFT_EYE,
            padding=15
        )

        right_eye, right_box = self._crop_region(
            frame,
            landmarks,
            self.RIGHT_EYE,
            padding=15
        )

        mouth, mouth_box = self._crop_region(
            frame,
            landmarks,
            self.MOUTH,
            padding=25
        )

        output["left_eye"] = {
            "crop": left_eye,
            "box": left_box
        }

        output["right_eye"] = {
            "crop": right_eye,
            "box": right_box
        }

        output["mouth"] = {
            "crop": mouth,
            "box": mouth_box
        }

        return output

    def close(self):

        self.face_mesh.close()