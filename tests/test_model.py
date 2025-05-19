import pytest
import torch

from src.model import Detector, FeatureExtractor


def test_feature_extractor(sample_input):
    model = FeatureExtractor()
    model.eval()

    with torch.no_grad():
        output = model(sample_input)

    batch_size = sample_input.size(0)
    assert output.shape == (batch_size, 2048, 7, 7), f"Unexpected shape: {output.shape}"


def test_detector_output_shape(detection_head, sample_input):
    backbone = FeatureExtractor()
    with torch.no_grad():
        features = backbone(sample_input)

    output = detection_head(features)
    batch_size = sample_input.size(0)
    assert output.shape == (batch_size, 7, 7, 30)  # 2*5 + 20 = 30


def test_full_yolo_model(sample_input):
    model = Detector()
    output = model(sample_input)
    batch_size = sample_input.size(0)
    assert output.shape == (batch_size, 7, 7, 30)


def test_detector_output_values_are_finite(detection_head):
    dummy_input = torch.randn(1, 2048, 7, 7)
    output = detection_head(dummy_input)
    assert torch.isfinite(output).all()


def test_iou():
    detector = Detector()
    box1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
    box2 = torch.tensor([1.0, 1.0, 3.0, 3.0])
    iou = detector._iou(box1, box2)
    assert torch.isclose(iou, torch.tensor(1 / 7), atol=1e-3)


def test_process_output_empty(detector):
    output = torch.zeros(7, 7, 5 * detector.num_boxes + detector.num_classes)
    boxes, scores, classes = detector._process_output(
        output, (448, 448), conf_threshold=0.99
    )
    assert boxes.shape[0] == 0
    assert scores.shape[0] == 0
    assert classes.shape[0] == 0


def test_predict_shape(detector):
    x = torch.randn(1, 3, 448, 448)
    detector.forward = lambda x: torch.zeros(
        1, 7, 7, 5 * detector.num_boxes + detector.num_classes
    )
    detector._process_output = lambda out, img_size, conf: (
        torch.zeros(0, 4),
        torch.zeros(0),
        torch.zeros(0),
    )
    detector.nms = lambda boxes, scores, classes, iou: []
    results = detector.predict(x)
    assert isinstance(results, list)
    assert len(results) == 1
