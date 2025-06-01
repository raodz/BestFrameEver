import os

import pytest

from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.utils.paths import KEY_PATH


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


@pytest.fixture
def vertex_config():
    return {
        "env_var_name": "GOOGLE_APPLICATION_CREDENTIALS",
        "credentials_path": KEY_PATH,
        "project_id": "bestframeever",
        "region": "us-central1",
        "model_name": "gemini-2.0-flash-001",
    }


@pytest.fixture
def prompt_template():
    return (
        "Give the names of the actors or actresses who played the {n_actors} main roles in the movie "
        '"{movie_title}".\nBe as concise as possible. Make sure you name the correct actors in the correct order.\n'
        "Only answer if you are confident in the correctness of the cast and the order of importance.\n"
        "If the movie does not exist or you are not sure about the correct cast, respond exactly with:\n"
        "Cannot find actors for this movie\nOtherwise, give your answer in the following format:\n"
        "First Name Last Name, First Name Last Name, First Name Last Name"
    )


HAS_KEY = os.path.exists(KEY_PATH)

skip_if_key = pytest.mark.skipif(
    HAS_KEY,
    reason="Vertex API key found — skipping mocked tests in favour of live API tests",
)

skip_if_no_key = pytest.mark.skipif(
    not HAS_KEY,
    reason="No local Vertex API key file found — skipping live API tests",
)
