import cv2

from inference.drowsiness_engine import DrowsinessEngine


print("=" * 60)
print("DROWSINESS ENGINE TEST")
print("=" * 60)

engine = DrowsinessEngine()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    engine.close()
    raise RuntimeError("Could not open webcam.")

print("\nCamera started.")
print("Press Q to quit.\n")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    result = engine.process(frame)

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    if result["face_detected"]:

        eye_state = result["eye_state"]
        yawn_state = result["yawn_state"]
        decision = result["decision"]

        print(
            f"\rEyes: {eye_state:<7} | "
            f"Yawn: {yawn_state:<7} | "
            f"Status: {decision['status']:<8} | "
            f"Eye closed: "
            f"{decision['eye_closure_duration']:.1f}s | "
            f"Yawns: {decision['yawn_count']}",
            end=""
        )

        # --------------------------------------
        # Draw eye boxes
        # --------------------------------------

        for eye_name in [
            "left_eye",
            "right_eye"
        ]:

            eye = result[eye_name]

            if eye is None:
                continue

            x1, y1, x2, y2 = eye["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{eye['state']} "
                f"{eye['confidence'] * 100:.0f}%",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        # --------------------------------------
        # Draw mouth box
        # --------------------------------------

        mouth = result["mouth"]

        if mouth is not None:

            x1, y1, x2, y2 = mouth["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 200, 0),
                2
            )

            cv2.putText(
                frame,
                f"{mouth['state']} "
                f"{mouth['confidence'] * 100:.0f}%",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1
            )

        # --------------------------------------
        # Status
        # --------------------------------------

        status = decision["status"]

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Drowsiness Engine Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
engine.close()
cv2.destroyAllWindows()

print("\n\n" + "=" * 60)
print("ENGINE TEST COMPLETE")
print("=" * 60)