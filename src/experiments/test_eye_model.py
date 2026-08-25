import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "eye_model_v1.pt"
)


# ==========================================
# LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Eye model not found:\n{MODEL_PATH}"
    )

print("Loading eye model...")

model = YOLO(str(MODEL_PATH))

print("Eye model loaded successfully.")


# ==========================================
# MEDIAPIPE FACE MESH
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# EYE LANDMARKS
# ==========================================

LEFT_EYE = [
    33, 133, 160, 159, 158,
    157, 173, 246, 161, 163,
    144, 145, 153, 154, 155
]

RIGHT_EYE = [
    362, 263, 387, 386, 385,
    384, 398, 466, 388, 390,
    373, 374, 380, 381, 382
]


# ==========================================
# COLORS (BGR)
# ==========================================

BG_COLOR = (15, 17, 23)
PANEL_COLOR = (25, 28, 36)
CARD_COLOR = (35, 39, 50)

TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (155, 160, 175)

GREEN = (80, 220, 120)
RED = (70, 70, 255)
ACCENT = (255, 170, 60)

CAMERA_BG = (8, 10, 14)


# ==========================================
# UI HELPER FUNCTIONS
# ==========================================

def rounded_rectangle(
    image,
    top_left,
    bottom_right,
    color,
    radius=20,
    thickness=-1
):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if thickness < 0:
        cv2.rectangle(
            image,
            (x1 + radius, y1),
            (x2 - radius, y2),
            color,
            -1
        )

        cv2.rectangle(
            image,
            (x1, y1 + radius),
            (x2, y2 - radius),
            color,
            -1
        )

        cv2.circle(
            image,
            (x1 + radius, y1 + radius),
            radius,
            color,
            -1
        )

        cv2.circle(
            image,
            (x2 - radius, y1 + radius),
            radius,
            color,
            -1
        )

        cv2.circle(
            image,
            (x1 + radius, y2 - radius),
            radius,
            color,
            -1
        )

        cv2.circle(
            image,
            (x2 - radius, y2 - radius),
            radius,
            color,
            -1
        )


def draw_text(
    image,
    text,
    position,
    scale,
    color,
    thickness=1
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ==========================================
# EYE CROP
# ==========================================

def get_eye_crop(frame, landmarks, indices):

    height, width = frame.shape[:2]

    points = []

    for idx in indices:

        x = int(landmarks[idx].x * width)
        y = int(landmarks[idx].y * height)

        points.append((x, y))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = min(xs)
    x2 = max(xs)

    y1 = min(ys)
    y2 = max(ys)

    eye_width = x2 - x1
    eye_height = y2 - y1

    padding_x = int(eye_width * 0.40)
    padding_y = int(eye_height * 1.0)

    x1 = max(x1 - padding_x, 0)
    x2 = min(x2 + padding_x, width)

    y1 = max(y1 - padding_y, 0)
    y2 = min(y2 + padding_y, height)

    eye_crop = frame[y1:y2, x1:x2]

    return eye_crop, (x1, y1, x2, y2)


# ==========================================
# MODEL PREDICTION
# ==========================================

def predict_eye(eye_crop):

    results = model.predict(
        eye_crop,
        imgsz=224,
        device="mps",
        verbose=False
    )

    result = results[0]

    class_id = result.probs.top1

    confidence = (
        result.probs.top1conf.item() * 100
    )

    class_name = (
        result.names[class_id].lower()
    )

    return class_name, confidence


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)


# ==========================================
# WINDOW
# ==========================================

WINDOW_NAME = "Drowsiness AI"

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 800

CAMERA_WIDTH = 940

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)


# ==========================================
# FPS
# ==========================================

previous_time = time.time()
fps = 0


print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Natural mirror view
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]

    # ======================================
    # MEDIAPIPE
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mesh_results = face_mesh.process(
        rgb_frame
    )


    # ======================================
    # DEFAULT VALUES
    # ======================================

    left_class = "waiting"
    right_class = "waiting"

    left_conf = 0
    right_conf = 0

    overall = "SEARCHING"
    overall_color = ACCENT

    face_detected = False


    # ======================================
    # FACE DETECTED
    # ======================================

    if mesh_results.multi_face_landmarks:

        face_detected = True

        landmarks = (
            mesh_results
            .multi_face_landmarks[0]
            .landmark
        )


        # ==================================
        # GET EYE CROPS
        # ==================================

        left_eye, left_box = get_eye_crop(
            frame,
            landmarks,
            LEFT_EYE
        )

        right_eye, right_box = get_eye_crop(
            frame,
            landmarks,
            RIGHT_EYE
        )


        if (
            left_eye.size > 0
            and right_eye.size > 0
        ):

            # ==============================
            # PREDICT
            # ==============================

            left_class, left_conf = (
                predict_eye(left_eye)
            )

            right_class, right_conf = (
                predict_eye(right_eye)
            )


            # ==============================
            # OVERALL RESULT
            # ==============================

            if (
                left_class == "closed"
                and right_class == "closed"
            ):

                overall = "CLOSED"
                overall_color = RED

            else:

                overall = "OPEN"
                overall_color = GREEN


            # ==============================
            # SUBTLE EYE BOXES
            # ==============================

            lx1, ly1, lx2, ly2 = left_box
            rx1, ry1, rx2, ry2 = right_box

            left_color = (
                RED
                if left_class == "closed"
                else GREEN
            )

            right_color = (
                RED
                if right_class == "closed"
                else GREEN
            )

            cv2.rectangle(
                frame,
                (lx1, ly1),
                (lx2, ly2),
                left_color,
                1
            )

            cv2.rectangle(
                frame,
                (rx1, ry1),
                (rx2, ry2),
                right_color,
                1
            )


    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    delta_time = (
        current_time - previous_time
    )

    if delta_time > 0:
        fps = 1 / delta_time

    previous_time = current_time


    # ======================================
    # CREATE UI CANVAS
    # ======================================

    ui = np.full(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3
        ),
        BG_COLOR,
        dtype=np.uint8
    )


    # ======================================
    # HEADER
    # ======================================

    draw_text(
        ui,
        "DROWSINESS AI",
        (35, 55),
        0.9,
        TEXT_PRIMARY,
        2
    )

    draw_text(
        ui,
        "REAL-TIME DRIVER MONITORING",
        (38, 85),
        0.45,
        TEXT_SECONDARY,
        1
    )


    # LIVE INDICATOR

    cv2.circle(
        ui,
        (CAMERA_WIDTH - 115, 48),
        7,
        RED,
        -1
    )

    draw_text(
        ui,
        "LIVE",
        (CAMERA_WIDTH - 95, 55),
        0.55,
        TEXT_PRIMARY,
        1
    )


    # ======================================
    # CAMERA CARD
    # ======================================

    camera_x = 25
    camera_y = 115

    camera_height = 650
    camera_panel_width = CAMERA_WIDTH - 50

    rounded_rectangle(
        ui,
        (camera_x, camera_y),
        (
            camera_x + camera_panel_width,
            camera_y + camera_height
        ),
        CAMERA_BG,
        radius=20
    )


    # ======================================
    # RESIZE CAMERA FRAME
    # ======================================

    available_width = camera_panel_width - 20
    available_height = camera_height - 20

    scale = min(
        available_width / frame_width,
        available_height / frame_height
    )

    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)

    resized_frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    frame_x = (
        camera_x
        + (camera_panel_width - new_width) // 2
    )

    frame_y = (
        camera_y
        + (camera_height - new_height) // 2
    )

    ui[
        frame_y:frame_y + new_height,
        frame_x:frame_x + new_width
    ] = resized_frame


    # ======================================
    # SIDEBAR
    # ======================================

    sidebar_x = CAMERA_WIDTH + 20

    # --------------------------------------
    # STATUS CARD
    # --------------------------------------

    rounded_rectangle(
        ui,
        (sidebar_x, 115),
        (DISPLAY_WIDTH - 25, 310),
        PANEL_COLOR,
        radius=20
    )

    draw_text(
        ui,
        "CURRENT EYE STATUS",
        (sidebar_x + 25, 155),
        0.45,
        TEXT_SECONDARY,
        1
    )

    draw_text(
        ui,
        overall,
        (sidebar_x + 25, 225),
        1.1,
        overall_color,
        2
    )

    cv2.circle(
        ui,
        (sidebar_x + 30, 265),
        6,
        GREEN if face_detected else RED,
        -1
    )

    tracking_text = (
        "FACE TRACKED"
        if face_detected
        else "SEARCHING FOR FACE"
    )

    draw_text(
        ui,
        tracking_text,
        (sidebar_x + 48, 270),
        0.42,
        TEXT_SECONDARY,
        1
    )


    # --------------------------------------
    # LEFT EYE CARD
    # --------------------------------------

    rounded_rectangle(
        ui,
        (sidebar_x, 335),
        (DISPLAY_WIDTH - 25, 455),
        CARD_COLOR,
        radius=16
    )

    draw_text(
        ui,
        "LEFT EYE",
        (sidebar_x + 20, 370),
        0.45,
        TEXT_SECONDARY,
        1
    )

    left_color = (
        RED
        if left_class == "closed"
        else GREEN
        if left_class == "open"
        else ACCENT
    )

    draw_text(
        ui,
        left_class.upper(),
        (sidebar_x + 20, 415),
        0.75,
        left_color,
        2
    )

    draw_text(
        ui,
        f"{left_conf:.1f}%",
        (DISPLAY_WIDTH - 120, 415),
        0.6,
        TEXT_PRIMARY,
        1
    )


    # --------------------------------------
    # RIGHT EYE CARD
    # --------------------------------------

    rounded_rectangle(
        ui,
        (sidebar_x, 475),
        (DISPLAY_WIDTH - 25, 595),
        CARD_COLOR,
        radius=16
    )

    draw_text(
        ui,
        "RIGHT EYE",
        (sidebar_x + 20, 510),
        0.45,
        TEXT_SECONDARY,
        1
    )

    right_color = (
        RED
        if right_class == "closed"
        else GREEN
        if right_class == "open"
        else ACCENT
    )

    draw_text(
        ui,
        right_class.upper(),
        (sidebar_x + 20, 555),
        0.75,
        right_color,
        2
    )

    draw_text(
        ui,
        f"{right_conf:.1f}%",
        (DISPLAY_WIDTH - 120, 555),
        0.6,
        TEXT_PRIMARY,
        1
    )


    # --------------------------------------
    # SYSTEM INFO
    # --------------------------------------

    rounded_rectangle(
        ui,
        (sidebar_x, 615),
        (DISPLAY_WIDTH - 25, 765),
        PANEL_COLOR,
        radius=16
    )

    draw_text(
        ui,
        "SYSTEM PERFORMANCE",
        (sidebar_x + 20, 650),
        0.42,
        TEXT_SECONDARY,
        1
    )

    draw_text(
        ui,
        f"{fps:.1f}",
        (sidebar_x + 20, 710),
        0.9,
        TEXT_PRIMARY,
        2
    )

    draw_text(
        ui,
        "FPS",
        (sidebar_x + 115, 708),
        0.45,
        TEXT_SECONDARY,
        1
    )

    draw_text(
        ui,
        "YOLO11 + MEDIAPIPE",
        (sidebar_x + 20, 745),
        0.4,
        ACCENT,
        1
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        WINDOW_NAME,
        ui
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()