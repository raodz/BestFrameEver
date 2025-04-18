import numpy as np
import pytest

from src.framesetgenerator import FrameSetGenerator
from src.movie import Movie


def test_frameset_initialization(sample_video_path):
    movie = Movie("Initmovie", actors=[])
    movie.read_video(sample_video_path)
    fsg = FrameSetGenerator(movie)

    assert fsg.movie == movie
    assert len(fsg) == 0


def test_get_all_frames_with_frequency_1(sample_video_path):
    movie = Movie("Freq1", actors=[])
    movie.read_video(sample_video_path)
    fsg = FrameSetGenerator(movie)
    fsg.get_all_frames(frequency=1)

    assert len(fsg) > 0
    assert all(isinstance(f, np.ndarray) for f in fsg)


def test_get_all_frames_with_frequency_5(sample_video_path):
    movie = Movie("Freq5", actors=[])
    movie.read_video(sample_video_path)
    fsg = FrameSetGenerator(movie)
    fsg.get_all_frames(frequency=5)

    total_frames = 0
    movie.read_video(sample_video_path)
    while movie.get_frame() is not None:
        total_frames += 1

    expected_max = total_frames // 5 + 1
    assert len(fsg) <= expected_max
    assert len(fsg) > 0


def test_get_all_frames_invalid_frequency_raises(sample_video_path):
    movie = Movie("InvalidFreq", actors=[])
    movie.read_video(sample_video_path)
    fsg = FrameSetGenerator(movie)

    with pytest.raises(ValueError):
        fsg.get_all_frames(frequency=0)


def test_indexing_frameset(sample_video_path):
    movie = Movie("IndexTest", actors=[])
    movie.read_video(sample_video_path)
    fsg = FrameSetGenerator(movie)
    fsg.get_all_frames(frequency=2)

    assert isinstance(fsg[0], np.ndarray)
    assert fsg[0].shape[0] > 0  # height
    assert fsg[0].shape[1] > 0  # width
