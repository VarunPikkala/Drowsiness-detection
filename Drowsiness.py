import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from ultralytics import YOLO


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR).
    """

    A = dist.euclidean(
        eye_landmarks[1],
        eye_landmarks[5]
    )

    B = dist.euclidean(
        eye_landmarks[2],
        eye_landmarks[4]
    )

    C = dist.euclidean(
        eye_landmarks[0],
        eye_landmarks[3]
    )

    if C == 0:
        return 1.0

    return (A + B) / (2.0 * C)


def detect_drowsiness():
    # ----------------------------------
    # YOLO
    # ----------------------------------
    print("Loading YOLO...")

    yolo_model = YOLO("yolov8n.pt")

    # ----------------------------------
    # MediaPipe
    # ----------------------------------
    print("Loading MediaPipe...")

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ----------------------------------
    # Eye landmarks
    # ----------------------------------
    LEFT_EYE = [
        362, 385, 387,
        263, 373, 380
    ]

    RIGHT_EYE = [
        33, 160, 158,
        133, 153, 144
    ]

    # ----------------------------------
    # Drowsiness settings
    # ----------------------------------

    # Try 0.20 first.
    # If false alarms happen, try 0.18.
    EAR_THRESHOLD = 0.20

    # Number of consecutive frames
    # with closed eyes before declaring
    # the person drowsy.
    CONSEC_FRAMES_THRESHOLD = 15

    drowsy_counter = 0

    # ----------------------------------
    # Camera
    # ----------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Camera started.")
    print("Press Q to quit.")

    previous_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        # ----------------------------------
        # FPS
        # ----------------------------------
        current_time = time.time()

        elapsed = current_time - previous_time

        fps = 1 / elapsed if elapsed > 0 else 0

        previous_time = current_time

        # ----------------------------------
        # Resize
        # ----------------------------------
        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        # ----------------------------------
        # YOLO person detection
        # ----------------------------------
        results = yolo_model(
            frame,
            classes=[0],
            conf=0.3,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Keep coordinates inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )

                person_roi = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_roi.size == 0:
                    continue

                # ----------------------------------
                # MediaPipe
                # ----------------------------------
                rgb_roi = cv2.cvtColor(
                    person_roi,
                    cv2.COLOR_BGR2RGB
                )

                face_results = face_mesh.process(
                    rgb_roi
                )

                if not face_results.multi_face_landmarks:
                    continue

                face_landmarks = (
                    face_results
                    .multi_face_landmarks[0]
                )

                # ----------------------------------
                # Get eye coordinates
                # ----------------------------------

                left_eye = []

                for index in LEFT_EYE:

                    landmark = (
                        face_landmarks
                        .landmark[index]
                    )

                    x = int(
                        landmark.x *
                        person_roi.shape[1]
                    )

                    y = int(
                        landmark.y *
                        person_roi.shape[0]
                    )

                    left_eye.append(
                        (x, y)
                    )

                right_eye = []

                for index in RIGHT_EYE:

                    landmark = (
                        face_landmarks
                        .landmark[index]
                    )

                    x = int(
                        landmark.x *
                        person_roi.shape[1]
                    )

                    y = int(
                        landmark.y *
                        person_roi.shape[0]
                    )

                    right_eye.append(
                        (x, y)
                    )

                # ----------------------------------
                # Calculate EAR
                # ----------------------------------

                left_ear = calculate_ear(
                    np.array(left_eye)
                )

                right_ear = calculate_ear(
                    np.array(right_eye)
                )

                ear = (
                    left_ear +
                    right_ear
                ) / 2.0

                # ----------------------------------
                # Drowsiness logic
                # ----------------------------------

                if ear < EAR_THRESHOLD:

                    drowsy_counter += 1

                else:

                    # Reset quickly when eyes open
                    drowsy_counter = 0

                if (
                    drowsy_counter
                    >= CONSEC_FRAMES_THRESHOLD
                ):

                    status = "DROWSY"

                    status_color = (
                        0,
                        0,
                        255
                    )

                else:

                    status = "ALERT"

                    status_color = (
                        0,
                        255,
                        0
                    )

                # ----------------------------------
                # Draw person box
                # ----------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    status_color,
                    2
                )

                # ----------------------------------
                # Status
                # ----------------------------------

                cv2.putText(
                    frame,
                    status,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )

                # ----------------------------------
                # Debug information
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Closed Frames: {drowsy_counter}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        # ----------------------------------
        # Show frame
        # ----------------------------------

        cv2.imshow(
            "Drowsiness Detection",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    face_mesh.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_drowsiness()