import torch

from src.dataset_preparing.face_dataset import FaceDataset
from utils import parse_bbox


def test_dataset_length(temp_images_and_annotations):
    image_dir, annotation_path = temp_images_and_annotations
    dataset = FaceDataset(images_root=image_dir, annotation_file=annotation_path)
    assert len(dataset) == 2


def test_dataset_returns_image_and_boxes(temp_images_and_annotations):
    image_dir, annotation_path = temp_images_and_annotations
    dataset = FaceDataset(images_root=image_dir, annotation_file=annotation_path)

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # RGB
    assert "boxes" in target
    assert isinstance(target["boxes"], torch.Tensor)
    assert target["boxes"].shape == (2, 4)  # 2 boksy, każdy x1,y1,x2,y2


def test_dataset_empty_boxes(temp_images_and_annotations):
    image_dir, annotation_path = temp_images_and_annotations
    dataset = FaceDataset(images_root=image_dir, annotation_file=annotation_path)

    _, target = dataset[1]
    assert target["boxes"].numel() == 0  # brak bboxów


def test_dataset_bbox_values(temp_images_and_annotations):
    image_dir, annotation_path = temp_images_and_annotations
    dataset = FaceDataset(images_root=image_dir, annotation_file=annotation_path)

    _, target = dataset[0]
    expected_boxes = torch.tensor(
        [parse_bbox(10, 20, 30, 40), parse_bbox(50, 60, 20, 10)], dtype=torch.float32
    )

    assert torch.allclose(target["boxes"], expected_boxes)
