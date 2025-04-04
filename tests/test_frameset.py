import pytest
import numpy as np
from src.Movie import Movie
from src.frameset import FrameSet


@pytest.fixture(params=[
    'sample_avi_video.avi',
    'sample_mp4_video.mp4'
])
def sample_video_path(request):
    return request.param


def test_frameset_initialization(sample_video_path):
    movie = Movie("InitMovie", [])
    movie.read_video(sample_video_path)
    fs = FrameSet(movie)

    assert fs.movie == movie
    assert len(fs) == 0


def test_get_all_frames_with_frequency_1(sample_video_path):
    movie = Movie("Freq1", [])
    movie.read_video(sample_video_path)
    fs = FrameSet(movie)
    fs.get_all_frames(frequency=1)

    assert len(fs) > 0
    assert all(isinstance(f, np.ndarray) for f in fs)


def test_get_all_frames_with_frequency_5(sample_video_path):
    movie = Movie("Freq5", [])
    movie.read_video(sample_video_path)
    fs = FrameSet(movie)
    fs.get_all_frames(frequency=5)

    total_frames = 0
    movie.read_video(sample_video_path)
    while movie.get_frame() is not None:
        total_frames += 1

    expected_max = total_frames // 5 + 1
    assert len(fs) <= expected_max
    assert len(fs) > 0


def test_get_all_frames_invalid_frequency_raises(sample_video_path):
    movie = Movie("InvalidFreq", [])
    movie.read_video(sample_video_path)
    fs = FrameSet(movie)

    with pytest.raises(ValueError):
        fs.get_all_frames(frequency=0)


def test_indexing_frameset(sample_video_path):
    movie = Movie("IndexTest", [])
    movie.read_video(sample_video_path)
    fs = FrameSet(movie)
    fs.get_all_frames(frequency=2)

    assert isinstance(fs[0], np.ndarray)
    assert fs[0].shape[0] > 0  # height
    assert fs[0].shape[1] > 0  # width
