import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "drowsiness_features_v1.pt"
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")


# ==========================================
# MEDIAPIPE FACE DETECTION
# ==========================================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)


# ==========================================
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)


# ==========================================
# UI CONFIGURATION
# ==========================================

WINDOW_NAME = "Drowsiness AI"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CAMERA_WIDTH = 900


CLASS_COLORS = {
    "open_eyes": (80, 220, 120),
    "closed_eyes": (70, 70, 255),
    "yawn": (0, 190, 255),
    "no_yawn": (220, 180, 80)
}


# ==========================================
# FPS
# ==========================================

previous_time = time.time()
fps = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # Original non-mirrored frame
    original_frame = frame.copy()

    frame_height, frame_width = frame.shape[:2]


    # ======================================
    # FACE DETECTION
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    detection_results = face_detection.process(
        rgb_frame
    )


    class_name = "no_face"
    confidence = 0.0
    probabilities = None
    inference_time = 0


    # ======================================
    # IF FACE IS DETECTED
    # ======================================

    if detection_results.detections:

        detection = detection_results.detections[0]

        bbox = (
            detection.location_data
            .relative_bounding_box
        )


        # Convert relative coordinates to pixels

        x = int(bbox.xmin * frame_width)
        y = int(bbox.ymin * frame_height)

        w = int(bbox.width * frame_width)
        h = int(bbox.height * frame_height)


        # Add padding around face

        padding_x = int(w * 0.20)
        padding_y = int(h * 0.20)


        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)

        x2 = min(
            frame_width,
            x + w + padding_x
        )

        y2 = min(
            frame_height,
            y + h + padding_y
        )


        # Crop face

        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        # Make sure crop is valid

        if face_crop.size > 0:

            # ==============================
            # YOLO PREDICTION
            # ==============================

            inference_start = time.time()

            results = model.predict(
                face_crop,
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
                probabilities
                .top1conf
                .item()
                * 100
            )

            class_name = (
                result.names[class_id]
                .lower()
                .replace(" ", "_")
            )


            # Draw face bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                CLASS_COLORS.get(
                    class_name,
                    (255, 255, 255)
                ),
                2
            )


    # ======================================
    # GET CLASS COLOR
    # ======================================

    color = CLASS_COLORS.get(
        class_name,
        (180, 180, 180)
    )


    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    elapsed_time = (
        current_time - previous_time
    )

    if elapsed_time > 0:
        fps = 1 / elapsed_time

    previous_time = current_time


    # ======================================
    # CREATE UI CANVAS
    # ======================================

    ui = np.zeros(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    # ======================================
    # CAMERA PANEL
    # ======================================

    frame_height, frame_width = frame.shape[:2]

    scale = min(
        CAMERA_WIDTH / frame_width,
        DISPLAY_HEIGHT / frame_height
    )

    new_width = int(
        frame_width * scale
    )

    new_height = int(
        frame_height * scale
    )

    resized_frame = cv2.resize(
        frame,
        (new_width, new_height)
    )


    x_offset = (
        CAMERA_WIDTH - new_width
    ) // 2

    y_offset = (
        DISPLAY_HEIGHT - new_height
    ) // 2


    ui[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized_frame


    # ======================================
    # SIDEBAR
    # ======================================

    cv2.rectangle(
        ui,
        (CAMERA_WIDTH, 0),
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        (25, 25, 25),
        -1
    )


    x_text = CAMERA_WIDTH + 30


    # Header

    cv2.putText(
        ui,
        "DROWSINESS",
        (x_text, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        ui,
        "AI FEATURE MONITORING",
        (x_text, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1
    )


    # ======================================
    # STATUS CARD
    # ======================================

    card_x = x_text
    card_y = 125

    card_width = 320
    card_height = 145


    cv2.rectangle(
        ui,
        (card_x, card_y),
        (
            card_x + card_width,
            card_y + card_height
        ),
        (45, 45, 45),
        -1
    )


    cv2.putText(
        ui,
        "CURRENT FEATURE",
        (
            card_x + 20,
            card_y + 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1
    )


    display_name = (
        class_name
        .replace("_", " ")
        .upper()
    )


    cv2.putText(
        ui,
        display_name,
        (
            card_x + 20,
            card_y + 85
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


    cv2.putText(
        ui,
        f"{confidence:.1f}% confidence",
        (
            card_x + 20,
            card_y + 120
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1
    )


    # ======================================
    # PROBABILITIES
    # ======================================

    cv2.putText(
        ui,
        "CLASS PROBABILITIES",
        (x_text, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


    y_text = 360


    if probabilities is not None:

        for i, prob in enumerate(
            probabilities.data.tolist()
        ):

            name = (
                result.names[i]
                .lower()
                .replace(" ", "_")
            )

            percentage = prob * 100

            bar_color = CLASS_COLORS.get(
                name,
                (255, 255, 255)
            )


            cv2.putText(
                ui,
                name
                .replace("_", " ")
                .upper(),
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1
            )


            cv2.putText(
                ui,
                f"{percentage:.1f}%",
                (x_text + 260, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


            bar_x = x_text
            bar_y = y_text + 15

            bar_width = 320
            bar_height = 16


            cv2.rectangle(
                ui,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),
                (65, 65, 65),
                -1
            )


            filled_width = int(
                bar_width * prob
            )


            cv2.rectangle(
                ui,
                (bar_x, bar_y),
                (
                    bar_x + filled_width,
                    bar_y + bar_height
                ),
                bar_color,
                -1
            )


            y_text += 70


    # ======================================
    # PERFORMANCE
    # ======================================

    cv2.putText(
        ui,
        f"FPS  {fps:.1f}",
        (x_text, 660),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    cv2.putText(
        ui,
        f"INFERENCE  {inference_time:.1f} ms",
        (x_text, 695),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        WINDOW_NAME,
        ui
    )


    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
face_detection.close()
cv2.destroyAllWindows()