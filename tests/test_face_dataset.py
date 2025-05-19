import pytest
import torch
from testing_paths import ANNOTATION_FILE, IMAGE_DIR

from src.dataset_preparing.face_dataset import FaceDataset
from utils import parse_bbox


@pytest.fixture
def face_dataset():
    return FaceDataset(images_root=IMAGE_DIR, annotation_file=ANNOTATION_FILE)


@pytest.fixture
def image_with_faces(face_dataset):
    return face_dataset[0]


@pytest.fixture
def image_without_faces(face_dataset):
    return face_dataset[1]


def test_dataset_length(face_dataset):
    assert len(face_dataset) == 2


def test_dataset_returns_image_and_boxes(image_with_faces):
    image, target = image_with_faces

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # RGB
    assert "boxes" in target
    assert isinstance(target["boxes"], torch.Tensor)
    assert target["boxes"].shape == (2, 4)


def test_dataset_empty_boxes(image_without_faces):
    _, target = image_without_faces
    assert target["boxes"].numel() == 0


def test_dataset_bbox_values(image_with_faces):
    _, target = image_with_faces

    actual_boxes = target["boxes"].tolist()

    expected_boxes = [parse_bbox(10, 20, 30, 40), parse_bbox(50, 60, 20, 10)]

    assert actual_boxes == expected_boxes
