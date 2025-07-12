import os

import pytest

from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.prediction.models.detector import Detector
from src.prediction.models.postprocessor import Postprocessor
from src.prediction.models.preprocessor import Preprocessor
from src.prediction.models.resnet50fe import ResNet50FE
from src.prediction.models.yolo_v1 import YoloV1
from src.utils.paths import KEY_PATH

has_key = os.path.exists(KEY_PATH)


@pytest.fixture(
    params=["testing_data/sample_avi_video.avi", "testing_data/sample_mp4_video.mp4"]
)
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
def default_preprocessor():
    return Preprocessor(
        input_size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale_factor=255.0,
    )


@pytest.fixture(params=[(2, 20, 7)])
def postprocessor(request):
    num_boxes, num_classes, grid_size = request.param
    return Postprocessor(
        num_boxes=num_boxes, num_classes=num_classes, grid_size=grid_size
    )


@pytest.fixture
def feature_extractor():
    return ResNet50FE(grid_size=7, output_feature_channels=2048)


@pytest.fixture
def detection_head(feature_extractor):
    return YoloV1(
        input_feature_channels=2048,
        grid_size=7,
        num_boxes=2,
        num_classes=20,
        hidden_size=4096,
        leaky_relu_negative_slope=0.1,
    )


@pytest.fixture
def detector(default_preprocessor, postprocessor, feature_extractor, detection_head):
    return Detector(
        input_size=(224, 224),
        num_boxes=2,
        num_classes=20,
        grid_size=7,
        conf_threshold=0.5,
        iou_threshold=0.5,
        feature_extractor_output_channels=2048,
        detection_head_input_feature_channels=2048,
        detection_head_hidden_size=4096,
        detection_head_leaky_relu_slope=0.1,
        preprocessor=default_preprocessor,
        postprocessor=postprocessor,
        device="cpu",
    )


skip_if_key = pytest.mark.skipif(
    has_key,
    reason="Vertex API key found — skipping mocked tests in favour of live API tests",
)

skip_if_no_key = pytest.mark.skipif(
    not has_key,
    reason="No local Vertex API key file found — skipping live API tests",
)
