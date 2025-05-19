import os

import pytest
import torch

from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.model import DetectionHead, Detector


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


@pytest.fixture(params=[1, 2, 4, 8, 16])
def sample_input(request):
    batch_size = request.param
    return torch.randn(batch_size, 3, 448, 448)


@pytest.fixture
def detection_head():
    return DetectionHead(num_boxes=2, num_classes=20)


@pytest.fixture
def detector():
    return Detector()
