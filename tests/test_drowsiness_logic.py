
import pytest

from logic.drowsiness_logic import DrowsinessLogic


@pytest.fixture
def logic():
    return DrowsinessLogic()


def test_initial_status_is_awake(logic, monkeypatch):
    monkeypatch.setattr("logic.drowsiness_logic.time.time", lambda: 100.0)

    result = logic.update("open", "no_yawn")

    assert result["status"] == "AWAKE"
    assert result["eye_closure_duration"] == 0.0
    assert result["yawn_count"] == 0
    assert result["yawn_active"] is False


def test_short_eye_closure_is_not_fatigue(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("closed", "no_yawn")
    current_time[0] = 100.3
    result = logic.update("closed", "no_yawn")

    assert result["eye_closure_duration"] == pytest.approx(0.3)
    assert result["status"] == "AWAKE"


def test_eye_closure_triggers_fatigue(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("closed", "no_yawn")
    current_time[0] = 101.5
    result = logic.update("closed", "no_yawn")

    assert result["status"] == "FATIGUED"
    assert result["eye_closure_duration"] == pytest.approx(1.5)


def test_long_eye_closure_triggers_drowsy(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("closed", "no_yawn")
    current_time[0] = 103.0
    result = logic.update("closed", "no_yawn")

    assert result["status"] == "DROWSY"
    assert result["eye_closure_duration"] == pytest.approx(3.0)


def test_open_eyes_reset_eye_closure_timer(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("closed", "no_yawn")
    current_time[0] = 101.0
    logic.update("closed", "no_yawn")

    result = logic.update("open", "no_yawn")

    assert result["eye_closure_duration"] == 0.0
    assert logic.eye_closed_start is None



def test_yawn_requires_confirmation_duration(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("open", "yawn")

    current_time[0] = 100.7
    result = logic.update("open", "yawn")

    assert result["yawn_count"] == 0
    assert result["yawn_active"] is False

    # Go slightly beyond the 0.8-second threshold
    current_time[0] = 100.81
    result = logic.update("open", "yawn")

    assert result["yawn_count"] == 1
    assert result["yawn_active"] is True



def test_continuous_yawn_is_counted_only_once(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.update("open", "yawn")

    current_time[0] = 101.0
    logic.update("open", "yawn")

    current_time[0] = 102.0
    result = logic.update("open", "yawn")

    assert result["yawn_count"] == 1



def test_three_confirmed_yawns_trigger_fatigue(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    for start in (100.0, 102.0, 104.0):
        current_time[0] = start
        logic.update("open", "yawn")

        current_time[0] = start + 0.81
        result = logic.update("open", "yawn")

        current_time[0] = start + 0.9
        logic.update("open", "no_yawn")

    assert result["yawn_count"] == 3
    assert result["status"] == "FATIGUED"



def test_old_yawns_expire_from_history(logic, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "logic.drowsiness_logic.time.time",
        lambda: current_time[0],
    )

    logic.yawn_history.extend([10.0, 20.0, 30.0])

    current_time[0] = 91.0
    result = logic.update("open", "no_yawn")

    assert result["yawn_count"] == 0


def test_reset_active_timers_preserves_yawn_history(logic):
    logic.eye_closed_start = 100.0
    logic.eye_closure_duration = 2.0
    logic.yawn_start = 101.0
    logic.yawn_active = True
    logic.yawn_history.append(99.0)

    logic.reset_active_timers()

    assert logic.eye_closed_start is None
    assert logic.eye_closure_duration == 0.0
    assert logic.yawn_start is None
    assert logic.yawn_active is False
    assert list(logic.yawn_history) == [99.0]
