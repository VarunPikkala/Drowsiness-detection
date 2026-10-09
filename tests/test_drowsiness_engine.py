
from unittest.mock import Mock

import numpy as np

from inference.drowsiness_engine import DrowsinessEngine


def make_engine():
    """Create an engine without loading real models or a webcam."""
    engine = DrowsinessEngine.__new__(DrowsinessEngine)

    engine.face_roi = Mock()
    engine.eye_classifier = Mock()
    engine.yawn_classifier = Mock()
    engine.logic = Mock()

    engine.logic.update.return_value = {
        "status": "AWAKE",
        "eye_closure_duration": 0.0,
        "yawn_count": 1,
        "yawn_active": False,
    }

    return engine


def test_missing_face_resets_active_timers():
    engine = make_engine()

    engine.face_roi.extract.return_value = {
        "left_eye": None,
        "right_eye": None,
        "mouth": None,
    }

    result = engine.process(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["face_detected"] is False
    assert result["decision"] is None

    engine.logic.reset_active_timers.assert_called_once()
    engine.logic.update.assert_not_called()
    engine.eye_classifier.predict.assert_not_called()
    engine.yawn_classifier.predict.assert_not_called()


def test_missing_eye_resets_active_timers():
    engine = make_engine()

    engine.face_roi.extract.return_value = {
        "left_eye": None,
        "right_eye": {"crop": np.zeros((10, 10, 3)), "box": (0, 0, 10, 10)},
        "mouth": {"crop": np.zeros((10, 10, 3)), "box": (0, 0, 10, 10)},
    }

    result = engine.process(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["face_detected"] is False
    engine.logic.reset_active_timers.assert_called_once()
    engine.logic.update.assert_not_called()


def test_valid_rois_run_classifiers_and_logic():
    engine = make_engine()

    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    box = (1, 2, 11, 12)

    engine.face_roi.extract.return_value = {
        "left_eye": {"crop": crop, "box": box},
        "right_eye": {"crop": crop, "box": box},
        "mouth": {"crop": crop, "box": box},
    }

    engine.eye_classifier.predict.return_value = {
        "class": "open",
        "confidence": 0.95,
        "probabilities": [0.05, 0.95],
    }

    engine.yawn_classifier.predict.return_value = {
        "class": "no_yawn",
        "confidence": 0.90,
        "probabilities": [0.90, 0.10],
    }

    result = engine.process(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["face_detected"] is True
    assert result["eye_state"] == "open"
    assert result["yawn_state"] == "no_yawn"
    assert result["decision"]["status"] == "AWAKE"

    assert engine.eye_classifier.predict.call_count == 2
    engine.yawn_classifier.predict.assert_called_once()
    engine.logic.update.assert_called_once_with("open", "no_yawn")
    engine.logic.reset_active_timers.assert_not_called()
