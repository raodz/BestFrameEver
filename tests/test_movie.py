import cv2
import numpy as np

from src.movie import Movie


def test_movie_initialization(tmp_path):
    fake_path = tmp_path / "test.mp4"
    movie = Movie(
        name="Test movie", file_path=str(fake_path), actors=["Actor A", "Actor B"]
    )
    assert movie.name == "Test movie"
    assert movie.file_path == str(fake_path)
    assert movie.actors == ["Actor A", "Actor B"]
    assert movie._cap is None


def test_cap_initialization_success(movie):
    assert isinstance(movie.cap, cv2.VideoCapture)
    assert movie.cap.isOpened()


def test_cap_initialization_failure(unloaded_movie):
    assert unloaded_movie.cap is None
    assert unloaded_movie._cap is None


def test_get_frame_returns_frame(movie):
    frame = movie.get_frame()
    assert frame is not None, "Expected a frame, but got None"
    assert isinstance(frame, np.ndarray), "Returned frame is not a NumPy array"

    movie.release()


def test_get_frame_no_video(unloaded_movie):
    frame = unloaded_movie.get_frame()
    assert frame is None, "Expected None when no video is opened"


def test_release_logs(caplog, movie):
    movie.cap
    with caplog.at_level("INFO"):
        movie.release()
        assert (
            "Video capture released" in caplog.text
        ), "Expected release log message not found"


def test_double_release_no_error(caplog, movie):
    movie.release()
    with caplog.at_level("INFO"):
        movie.release()
        assert (
            "already released" in caplog.text or "not initialized" in caplog.text
        ), "Expected graceful double release message"
