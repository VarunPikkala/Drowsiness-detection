import cv2

from vision.face_roi import FaceROI


print("=" * 50)
print("LIVE FACE ROI TEST")
print("=" * 50)

roi_detector = FaceROI()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    rois = roi_detector.extract(frame)

    # ------------------------------------------
    # Draw ROIs
    # ------------------------------------------

    for name in ["left_eye", "right_eye", "mouth"]:

        roi = rois[name]

        if roi is None:
            continue

        x1, y1, x2, y2 = roi["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow(
        "Face ROI Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
roi_detector.close()
cv2.destroyAllWindows()

print("ROI TEST COMPLETE")