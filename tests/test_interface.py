import pytest
from unittest.mock import MagicMock, patch
from App import Interface   # << dein Modul importieren
import cv2
import numpy as np


@pytest.fixture
def mock_robot_comm():
    mock = MagicMock()
    mock.robot_states = {"r1": "idle", "r2": "idle"}
    mock.robot_comm_status = "ok"
    return mock


@pytest.fixture
def interface(mock_robot_comm):
    return Interface(mock_robot_comm)


@pytest.fixture
def client(interface):
    interface.app.config["TESTING"] = True
    return interface.app.test_client()


# ------------------------------
# Test: INDEX ROUTE
# ------------------------------
def test_index_route(client, mock_robot_comm):
    response = client.get("/")
    assert response.status_code == 200
    # Template wird korrekt gerendert, States übergeben
    assert b"idle" in response.data


# ------------------------------
# Test: START ROUTE
# ------------------------------
def test_start_route(client, mock_robot_comm):
    response = client.get("/start", follow_redirects=True)
    assert response.status_code == 200

    # Alle Roboter sollten auf "peleando" gesetzt sein
    assert all(state == "peleando" for state in mock_robot_comm.robot_states.values())

    # Log wurde aufgerufen
    mock_robot_comm.log.assert_called_with(
        "PELEA →", "Se inició la pelea"
    )


# ------------------------------
# Test: STOP ROUTE
# ------------------------------
def test_stop_route(client, mock_robot_comm):
    response = client.get("/stop", follow_redirects=True)
    assert response.status_code == 200

    assert all(state == "fuera de combate" for state in mock_robot_comm.robot_states.values())

    mock_robot_comm.log.assert_called_with(
        "PELEA →", "Se detuvo la pelea"
    )


# ------------------------------
# Test: gen_frames erzeugt korrekte JPG-Daten
# ------------------------------
def test_gen_frames(interface):
    # Dummy-Frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    interface.update_frame(dummy_frame)

    # Patch cv2.imencode, damit kein echtes JPEG erzeugt werden muss
    with patch("cv2.imencode", return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
        gen = interface.gen_frames()
        frame_chunk = next(gen)

    assert b"--frame" in frame_chunk
    assert b"Content-Type: image/jpeg" in frame_chunk
    assert b"\r\n1\x00\x00\x00"  # Output beginnt mit den Bytes aus unserem Mock-Array
