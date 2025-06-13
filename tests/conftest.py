import os

import pytest

from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.utils.paths import KEY_PATH

has_key = os.path.exists(KEY_PATH)


@pytest.fixture(params=["sample_avi_video.avi", "sample_mp4_video.mp4"])
def sample_video_path(request):
    return os.path.join(os.path.dirname(__file__), request.param)


@pytest.fixture
def movie(sample_video_path):
    movie = Movie("Freq1", sample_video_path, actors=[])
    return movie


@pytest.fixture
def unloaded_movie(tmp_path):
    fake_path = tmp_path / "unloaded.mp4"
    return Movie("Unloaded", str(fake_path), actors=[])


@pytest.fixture
def flc(movie):
    return FramesListCreator(movie)


skip_if_key = pytest.mark.skipif(
    has_key,
    reason="Vertex API key found — skipping mocked tests in favour of live API tests",
)

skip_if_no_key = pytest.mark.skipif(
    not has_key,
    reason="No local Vertex API key file found — skipping live API tests",
)
