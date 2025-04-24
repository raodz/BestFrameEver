import os

import pytest

from src.frames_list_creator import FramesListCreator
from src.movie import Movie


@pytest.fixture(params=["sample_avi_video.avi", "sample_mp4_video.mp4"])
def sample_video_path(request):
    return os.path.join(os.path.dirname(__file__), request.param)


@pytest.fixture
def movie(sample_video_path):
    movie = Movie("Freq1", actors=[])
    movie.read_video(sample_video_path)
    return movie


@pytest.fixture
def unloaded_movie():
    return Movie("Unloaded", actors=[])


@pytest.fixture
def frame_set_creator(movie):
    return FramesListCreator(movie)
