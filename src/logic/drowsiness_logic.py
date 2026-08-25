import time
from collections import deque


class DrowsinessLogic:

    def __init__(self):

        # ======================================
        # EYE CLOSURE SETTINGS
        # ======================================

        # Short closures are normal blinks
        self.BLINK_MAX_DURATION = 0.4

        # Sustained closure indicates fatigue
        self.FATIGUE_EYE_DURATION = 1.5

        # Long closure indicates drowsiness
        self.DROWSY_EYE_DURATION = 3.0

        self.eye_closed_start = None
        self.eye_closure_duration = 0.0


        # ======================================
        # YAWN SETTINGS
        # ======================================

        # Yawn must persist this long
        # before being counted as confirmed
        self.YAWN_CONFIRM_DURATION = 0.8

        self.yawn_start = None
        self.yawn_active = False

        # Store timestamps of confirmed yawns
        self.yawn_history = deque()

        # Count yawns within this time window
        self.YAWN_WINDOW = 60


    # ==========================================
    # UPDATE SYSTEM
    # ==========================================

    def update(self, eye_state, yawn_state):

        current_time = time.time()

        # ======================================
        # EYE TEMPORAL ANALYSIS
        # ======================================

        if eye_state == "closed":

            if self.eye_closed_start is None:

                self.eye_closed_start = current_time

            self.eye_closure_duration = (
                current_time - self.eye_closed_start
            )

        else:

            self.eye_closed_start = None
            self.eye_closure_duration = 0.0


        # ======================================
        # YAWN TEMPORAL ANALYSIS
        # ======================================

        if yawn_state == "yawn":

            if self.yawn_start is None:

                self.yawn_start = current_time

            yawn_duration = (
                current_time - self.yawn_start
            )

            # Confirm only once
            if (
                yawn_duration >= self.YAWN_CONFIRM_DURATION
                and not self.yawn_active
            ):

                self.yawn_history.append(
                    current_time
                )

                self.yawn_active = True

        else:

            self.yawn_start = None
            self.yawn_active = False


        # ======================================
        # REMOVE OLD YAWNS
        # ======================================

        while (
            self.yawn_history
            and current_time - self.yawn_history[0]
            > self.YAWN_WINDOW
        ):

            self.yawn_history.popleft()


        yawn_count = len(
            self.yawn_history
        )


        # ======================================
        # DECISION SYSTEM
        # ======================================

        if (
            self.eye_closure_duration
            >= self.DROWSY_EYE_DURATION
        ):

            status = "DROWSY"

        elif (
            self.eye_closure_duration
            >= self.FATIGUE_EYE_DURATION
            or yawn_count >= 3
        ):

            status = "FATIGUED"

        else:

            status = "AWAKE"


        # ======================================
        # RETURN RESULTS
        # ======================================

        return {
            "status": status,
            "eye_closure_duration":
                self.eye_closure_duration,

            "yawn_count": yawn_count,

            "yawn_active":
                self.yawn_active
        }