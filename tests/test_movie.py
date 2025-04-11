import pytest
import numpy as np
from src.Movie import Movie
import os


@pytest.fixture(params=[
    'sample_avi_video.avi',
    'sample_mp4_video.mp4'
])
def sample_video_path(request):
    return request.param


def test_movie_initialization(sample_video_path):
    movie = Movie(name="Test Movie", actors=["Actor A", "Actor B"])
    assert movie.name == "Test Movie"
    assert movie.actors == ["Actor A", "Actor B"]
    assert movie.cap is None


def test_read_video_success(sample_video_path):
    assert os.path.isfile(
        sample_video_path), f"Plik nie istnieje: {sample_video_path}"
    movie = Movie("Test Movie", [])
    result = movie.read_video(sample_video_path)
    assert result is True
    assert movie.cap is not None
    assert movie.cap.isOpened()


def test_read_video_failure(tmp_path):
    fake_path = tmp_path / "not_a_video.mp4"
    fake_path.write_text("not really a video")
    movie = Movie("Bad Movie", [])
    result = movie.read_video(str(fake_path))
    assert result is False
    assert movie.cap is not None
    assert not movie.cap.isOpened()


def test_get_frame_returns_frame(sample_video_path):
    movie = Movie("Frame Test", [])
    movie.read_video(sample_video_path)

    frame = movie.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)

    movie.release()


def test_get_frame_no_video():
    movie = Movie("Empty", [])
    frame = movie.get_frame()
    assert frame is None


def test_release_logs(caplog, sample_video_path):
    movie = Movie("Release Test", [])
    movie.read_video(sample_video_path)

    with caplog.at_level("INFO"):
        movie.release()
        assert "Video capture released" in caplog.text


def test_double_release_no_error(caplog, sample_video_path):
    movie = Movie("Double Release", [])
    movie.read_video(sample_video_path)

    movie.release()
    with caplog.at_level("INFO"):
        movie.release()
        assert "already released" in caplog.text
