import cv2
import time
from pathlib import Path

import mediapipe as mp
from ultralytics import YOLO


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "yawn_model_v1.pt"
)


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading yawn model...")

model = YOLO(str(MODEL_PATH))

print("Loading face detector...")

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

print("Models loaded successfully.")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


# ==========================================
# CONFIGURATION
# ==========================================

WINDOW_NAME = "Yawn Detection AI"

COLORS = {
    "no_yawn": (80, 220, 120),
    "yawn": (60, 80, 255),
}

previous_time = time.time()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]


    # ======================================
    # FACE DETECTION
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    detection_results = face_detector.process(
        rgb_frame
    )


    display_frame = frame.copy()

    class_name = "no face"
    confidence = 0
    inference_time = 0


    # ======================================
    # IF FACE FOUND
    # ======================================

    if detection_results.detections:

        detection = detection_results.detections[0]

        bbox = (
            detection.location_data.relative_bounding_box
        )


        # Face coordinates
        x = int(bbox.xmin * frame_width)
        y = int(bbox.ymin * frame_height)

        w = int(bbox.width * frame_width)
        h = int(bbox.height * frame_height)


        # Keep coordinates inside frame
        x = max(0, x)
        y = max(0, y)

        x2 = min(frame_width, x + w)
        y2 = min(frame_height, y + h)


        # ==================================
        # MOUTH REGION
        # ==================================
        # Lower-middle section of face

        mouth_y1 = y + int(h * 0.50)
        mouth_y2 = y + int(h * 0.95)

        mouth_x1 = x + int(w * 0.10)
        mouth_x2 = x + int(w * 0.90)


        mouth_y1 = max(0, mouth_y1)
        mouth_y2 = min(frame_height, mouth_y2)

        mouth_x1 = max(0, mouth_x1)
        mouth_x2 = min(frame_width, mouth_x2)


        mouth_region = frame[
            mouth_y1:mouth_y2,
            mouth_x1:mouth_x2
        ]


        # ==================================
        # YOLO PREDICTION
        # ==================================

        if mouth_region.size > 0:

            inference_start = time.time()

            results = model.predict(
                mouth_region,
                imgsz=224,
                device="mps",
                verbose=False
            )

            inference_time = (
                time.time()
                - inference_start
            ) * 1000


            result = results[0]

            probabilities = result.probs

            class_id = probabilities.top1

            confidence = (
                probabilities.top1conf.item()
                * 100
            )

            class_name = (
                result.names[class_id].lower()
            )


            # ==================================
            # DRAW FACE BOX
            # ==================================

            color = COLORS.get(
                class_name,
                (255, 255, 255)
            )

            cv2.rectangle(
                display_frame,
                (x, y),
                (x2, y2),
                color,
                2
            )


            # ==================================
            # DRAW MOUTH REGION
            # ==================================

            cv2.rectangle(
                display_frame,
                (mouth_x1, mouth_y1),
                (mouth_x2, mouth_y2),
                color,
                2
            )


            cv2.putText(
                display_frame,
                "MOUTH REGION",
                (
                    mouth_x1,
                    mouth_y1 - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )


    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    # ======================================
    # MODERN UI PANEL
    # ======================================

    overlay = display_frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame_width, 125),
        (20, 20, 20),
        -1
    )

    display_frame = cv2.addWeighted(
        overlay,
        0.70,
        display_frame,
        0.30,
        0
    )


    # Title
    cv2.putText(
        display_frame,
        "YAWN DETECTION AI",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # Status
    status_color = COLORS.get(
        class_name,
        (200, 200, 200)
    )

    cv2.putText(
        display_frame,
        class_name.upper(),
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # Confidence
    cv2.putText(
        display_frame,
        f"{confidence:.1f}%",
        (300, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # PERFORMANCE INFO
    # ======================================

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (30, frame_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display_frame,
        f"Inference: {inference_time:.1f} ms",
        (30, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

face_detector.close()

cv2.destroyAllWindows()