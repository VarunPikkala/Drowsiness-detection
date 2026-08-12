import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.20
CONSEC_FRAMES_THRESHOLD = 15

# Take one accuracy sample every N seconds
EVALUATION_INTERVAL = 1.0


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(eye_landmarks):
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading YOLO model...")
    yolo_model = YOLO("yolov8n.pt")

    print("Initializing MediaPipe...")

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Eye landmarks
    LEFT_EYE = [
        362, 385, 387,
        263, 373, 380
    ]

    RIGHT_EYE = [
        33, 160, 158,
        133, 153, 144
    ]

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("\nCamera started.")
    print("A = Actually Awake")
    print("D = Actually Drowsy")
    print("SPACE = Reset accuracy")
    print("Q = Quit\n")

    # ========================================================
    # VARIABLES
    # ========================================================

    drowsy_counter = 0

    previous_time = time.time()
    last_evaluation_time = time.time()

    # Accuracy statistics
    total_samples = 0
    correct_samples = 0

    # Confusion matrix
    true_awake = 0
    false_drowsy = 0

    true_drowsy = 0
    false_awake = 0

    # User's current ground truth
    actual_state = None

    # Current prediction
    predicted_state = "UNKNOWN"

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        elapsed = current_time - previous_time

        fps = 1 / elapsed if elapsed > 0 else 0

        previous_time = current_time

        # ====================================================
        # RESIZE
        # ====================================================

        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        # ====================================================
        # YOLO PERSON DETECTION
        # ====================================================

        results = yolo_model(
            frame,
            classes=[0],
            conf=0.3,
            verbose=False
        )

        face_found = False
        current_ear = 1.0

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Keep coordinates inside image
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

                # =================================================
                # MEDIAPIPE FACE MESH
                # =================================================

                rgb_roi = cv2.cvtColor(
                    person_roi,
                    cv2.COLOR_BGR2RGB
                )

                face_results = face_mesh.process(
                    rgb_roi
                )

                if not face_results.multi_face_landmarks:
                    continue

                face_found = True

                face_landmarks = (
                    face_results.multi_face_landmarks[0]
                )

                # =================================================
                # LEFT EYE
                # =================================================

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

                # =================================================
                # RIGHT EYE
                # =================================================

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

                # =================================================
                # EAR
                # =================================================

                try:

                    left_ear = calculate_ear(
                        np.array(left_eye)
                    )

                    right_ear = calculate_ear(
                        np.array(right_eye)
                    )

                    current_ear = (
                        left_ear +
                        right_ear
                    ) / 2.0

                except Exception:

                    current_ear = 1.0

                # =================================================
                # DROWSINESS LOGIC
                # =================================================

                if current_ear < EAR_THRESHOLD:

                    drowsy_counter += 1

                else:

                    drowsy_counter = 0

                if (
                    drowsy_counter
                    >= CONSEC_FRAMES_THRESHOLD
                ):

                    predicted_state = "DROWSY"

                else:

                    predicted_state = "AWAKE"

                # =================================================
                # DRAW BOX
                # =================================================

                if predicted_state == "DROWSY":

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    predicted_state,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )

                # We only need the first detected person
                break

            if face_found:
                break

        # ========================================================
        # REAL-TIME EVALUATION
        # ========================================================

        if (
            actual_state is not None
            and face_found
            and current_time - last_evaluation_time
            >= EVALUATION_INTERVAL
        ):

            # Compare model prediction with ground truth
            if predicted_state == actual_state:

                correct_samples += 1

                if actual_state == "AWAKE":
                    true_awake += 1
                else:
                    true_drowsy += 1

            else:

                if actual_state == "AWAKE":
                    false_drowsy += 1
                else:
                    false_awake += 1

            total_samples += 1

            last_evaluation_time = current_time

        # ========================================================
        # ACCURACY
        # ========================================================

        if total_samples > 0:

            accuracy = (
                correct_samples /
                total_samples
            ) * 100

        else:

            accuracy = 0.0

        # ========================================================
        # PRECISION
        # ========================================================

        precision_denominator = (
            true_drowsy +
            false_drowsy
        )

        if precision_denominator > 0:

            precision = (
                true_drowsy /
                precision_denominator
            ) * 100

        else:

            precision = 0.0

        # ========================================================
        # RECALL
        # ========================================================

        recall_denominator = (
            true_drowsy +
            false_awake
        )

        if recall_denominator > 0:

            recall = (
                true_drowsy /
                recall_denominator
            ) * 100

        else:

            recall = 0.0

        # ========================================================
        # F1
        # ========================================================

        if precision + recall > 0:

            f1 = (
                2 * precision * recall
            ) / (precision + recall)

        else:

            f1 = 0.0

        # ========================================================
        # DISPLAY
        # ========================================================

        cv2.putText(
            frame,
            f"Prediction: {predicted_state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if actual_state is not None:

            cv2.putText(
                frame,
                f"Actual: {actual_state}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Actual: PRESS A or D",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"EAR: {current_ear:.3f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy:.2f}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {total_samples}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Precision: {precision:.2f}%",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Recall: {recall:.2f}%",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"F1: {f1:.2f}%",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ========================================================
        # INSTRUCTIONS
        # ========================================================

        cv2.putText(
            frame,
            "A = Awake | D = Drowsy | SPACE = Reset | Q = Quit",
            (20, 700),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ========================================================
        # SHOW
        # ========================================================

        cv2.imshow(
            "Real-Time Drowsiness Evaluation",
            frame
        )

        # ========================================================
        # KEYBOARD
        # ========================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):

            actual_state = "AWAKE"

            print("Ground truth set to: AWAKE")

        elif key == ord("d"):

            actual_state = "DROWSY"

            print("Ground truth set to: DROWSY")

        elif key == ord(" "):

            total_samples = 0
            correct_samples = 0

            true_awake = 0
            false_drowsy = 0
            true_drowsy = 0
            false_awake = 0

            print("Accuracy statistics reset.")

        elif key == ord("q"):

            break

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(
        f"Total samples : {total_samples}"
    )

    print(
        f"Correct       : {correct_samples}"
    )

    print(
        f"Accuracy      : {accuracy:.2f}%"
    )

    print(
        f"Precision     : {precision:.2f}%"
    )

    print(
        f"Recall        : {recall:.2f}%"
    )

    print(
        f"F1 Score      : {f1:.2f}%"
    )

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Awake   Drowsy")

    print(
        f"Actual Awake    {true_awake:6d}"
        f" {false_drowsy:8d}"
    )

    print(
        f"Actual Drowsy   {false_awake:6d}"
        f" {true_drowsy:8d}"
    )

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()