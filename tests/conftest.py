import os
import tempfile

import cv2
import numpy as np
import pytest

from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie


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


sample_annotation_conent = """test1.jpg
2
10 20 30 40 0 0 0 0 0 0
50 60 20 10 0 0 0 0 0 0
test2.jpg
0
"""


@pytest.fixture
def temp_images_and_annotations():
    with tempfile.TemporaryDirectory() as tmpdir:
        for img_name in ["test1.jpg", "test2.jpg"]:
            img_path = os.path.join(tmpdir, img_name)
            dummy_image = np.full((100, 100, 3), 255, dtype=np.uint8)
            cv2.imwrite(img_path, dummy_image)

        annotation_path = os.path.join(tmpdir, "annotations.txt")
        with open(annotation_path, "w") as f:
            f.write(sample_annotation_conent)

        yield tmpdir, annotation_path
