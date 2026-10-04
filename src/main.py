import cv2
import time

from inference.drowsiness_engine import DrowsinessEngine


# ==========================================
# CONFIGURATION
# ==========================================

WINDOW_NAME = "Drowsiness Detection AI"

STATUS_COLORS = {
    "AWAKE": (80, 220, 120),
    "FATIGUED": (0, 180, 255),
    "DROWSY": (60, 80, 255),
}

DETECTOR_COLORS = {
    "open": (80, 220, 120),
    "closed": (60, 80, 255),
    "yawn": (0, 180, 255),
    "no_yawn": (80, 220, 120),
    "unknown": (180, 180, 180),
}


# ==========================================
# LOAD AI SYSTEM
# ==========================================

print("==========================================")
print("   DROWSINESS DETECTION AI")
print("==========================================")
print()
print("Loading AI system...")

engine = DrowsinessEngine()

print()
print("System ready.")
print()


# ==========================================
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    engine.close()
    raise RuntimeError("Could not open webcam.")


# Raspberry Pi 4 target configuration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


# ==========================================
# FPS
# ==========================================

previous_time = time.time()


# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to read frame from webcam.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)


        # ======================================
        # RUN DROWSINESS ENGINE
        # ======================================

        inference_start = time.time()

        result = engine.process(frame)

        inference_time = (
            time.time() - inference_start
        ) * 1000


        # ======================================
        # FPS CALCULATION
        # ======================================

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.001
        )

        previous_time = current_time


        # ======================================
        # DEFAULT STATES
        # ======================================

        system_status = "AWAKE"
        eye_state = "unknown"
        yawn_state = "unknown"

        eye_closure_duration = 0.0
        yawn_count = 0

        status_color = STATUS_COLORS["AWAKE"]
        eye_color = DETECTOR_COLORS["unknown"]
        yawn_color = DETECTOR_COLORS["unknown"]


        # ======================================
        # FACE DETECTED
        # ======================================

        if result["face_detected"]:

            # ----------------------------------
            # EYE INFORMATION
            # ----------------------------------

            eye_state = result["eye_state"]

            if result["left_eye"] is not None:

                left_eye = result["left_eye"]

                x1, y1, x2, y2 = left_eye["box"]

                state = left_eye["state"]
                confidence = left_eye["confidence"]

                color = DETECTOR_COLORS.get(
                    state,
                    DETECTOR_COLORS["unknown"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{state.upper()} {confidence * 100:.0f}%",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )


            # ----------------------------------
            # RIGHT EYE
            # ----------------------------------

            if result["right_eye"] is not None:

                right_eye = result["right_eye"]

                x1, y1, x2, y2 = right_eye["box"]

                state = right_eye["state"]
                confidence = right_eye["confidence"]

                color = DETECTOR_COLORS.get(
                    state,
                    DETECTOR_COLORS["unknown"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{state.upper()} {confidence * 100:.0f}%",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )


            # ----------------------------------
            # MOUTH
            # ----------------------------------

            if result["mouth"] is not None:

                mouth = result["mouth"]

                x1, y1, x2, y2 = mouth["box"]

                yawn_state = mouth["state"]
                confidence = mouth["confidence"]

                yawn_color = DETECTOR_COLORS.get(
                    yawn_state,
                    DETECTOR_COLORS["unknown"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    yawn_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{yawn_state.upper()} {confidence * 100:.0f}%",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    yawn_color,
                    1
                )


            # ----------------------------------
            # DECISION
            # ----------------------------------

            decision = result["decision"]

            if decision is not None:

                system_status = decision["status"]

                eye_closure_duration = (
                    decision["eye_closure_duration"]
                )

                yawn_count = decision["yawn_count"]


            # ----------------------------------
            # COLORS
            # ----------------------------------

            status_color = STATUS_COLORS.get(
                system_status,
                (255, 255, 255)
            )

            eye_color = DETECTOR_COLORS.get(
                eye_state,
                DETECTOR_COLORS["unknown"]
            )


        # ======================================
        # NO FACE DETECTED
        # ======================================

        else:

            system_status = "NO FACE"

            status_color = (180, 180, 180)

            eye_state = "unknown"
            yawn_state = "unknown"

            eye_closure_duration = 0.0
            yawn_count = 0


        # ======================================
        # TOP DASHBOARD
        # ======================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], 145),
            (18, 18, 18),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.78,
            frame,
            0.22,
            0
        )


        # ======================================
        # TITLE
        # ======================================

        cv2.putText(
            frame,
            "DROWSINESS DETECTION AI",
            (30, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ======================================
        # SYSTEM STATUS
        # ======================================

        cv2.putText(
            frame,
            "SYSTEM STATUS",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1
        )

        cv2.putText(
            frame,
            system_status,
            (30, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )


        # ======================================
        # EYE STATUS
        # ======================================

        cv2.putText(
            frame,
            "EYES",
            (260, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1
        )

        cv2.putText(
            frame,
            eye_state.upper(),
            (260, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            eye_color,
            2
        )


        # ======================================
        # MOUTH STATUS
        # ======================================

        cv2.putText(
            frame,
            "MOUTH",
            (430, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1
        )

        cv2.putText(
            frame,
            yawn_state.upper(),
            (430, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            yawn_color,
            2
        )


        # ======================================
        # EYE CLOSURE TIMER
        # ======================================

        cv2.putText(
            frame,
            f"Eye closure: {eye_closure_duration:.1f}s",
            (30, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )


        # ======================================
        # YAWN COUNTER
        # ======================================

        cv2.putText(
            frame,
            f"Confirmed yawns: {yawn_count}",
            (260, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )


        # ======================================
        # PERFORMANCE INFO
        # ======================================

        height = frame.shape[0]

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (25, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Inference: {inference_time:.1f} ms",
            (25, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


        # ======================================
        # DISPLAY
        # ======================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ==========================================
# CLEANUP
# ==========================================

finally:

    cap.release()

    engine.close()

    cv2.destroyAllWindows()

    print()
    print("Drowsiness Detection AI stopped.")