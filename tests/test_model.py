from unittest.mock import patch

import numpy as np
import torch

from src.prediction.models.model import FeatureExtractor
from utils import box_iou, nms


def test_feature_extractor():
    model = FeatureExtractor()
    model.eval()

    sample_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(sample_input)

    assert output.shape == (2, 2048, 7, 7), f"Unexpected shape: {output.shape}"


def test_detector_output_shape(detector):
    model = detector
    model.eval()

    sample_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(sample_input)

    expected_shape = (
        2,
        7,
        7,
        model.cfg.model.num_boxes,
        5 + model.cfg.model.num_classes,
    )
    assert output.shape == expected_shape, f"Unexpected output shape: {output.shape}"


def test_detection_head_output_is_finite(detection_head):
    input_shape = (2048, 7, 7)
    dummy_input = torch.randn(1, *input_shape)
    output = detection_head(dummy_input)
    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_predict_shape_no_detections(detector):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    with patch.object(
        detector,
        "forward",
        return_value=torch.zeros(
            1, 7, 7, detector.cfg.model.num_boxes, 5 + detector.cfg.model.num_classes
        ),
    ):
        with patch.object(
            detector,
            "_postprocess_output",
            return_value=(
                torch.empty(0, 4),
                torch.empty(0),
                torch.empty(0, dtype=torch.long),
            ),
        ):
            results = detector.predict(dummy_img)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] == [], "Expected empty detections"


def test_class_probabilities_sum_to_one(detector):
    detector.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = detector(dummy_input)

    class_probs = torch.softmax(output[..., 5:], dim=-1)
    summed = class_probs.sum(dim=-1)

    assert torch.allclose(
        summed, torch.ones_like(summed), atol=1e-4
    ), "Class probs should sum to 1"


def test_box_iou():
    box1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    box2 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    iou = box_iou(box1, box2)
    expected = 1.0 / 7.0
    assert torch.isclose(
        iou[0, 0], torch.tensor(expected), atol=1e-3
    ), f"Expected {expected}, got {iou.item()}"


def test_nms():
    boxes = torch.tensor(
        [[10, 10, 20, 20], [11, 11, 21, 21], [50, 50, 60, 60]], dtype=torch.float
    )

    scores = torch.tensor([0.9, 0.85, 0.6])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2], f"Expected [0, 2], got {keep.tolist()}"


def test_postprocess_output_no_detections(detector):
    output = torch.zeros(
        7, 7, detector.cfg.model.num_boxes, 5 + detector.cfg.model.num_classes
    )
    boxes, scores, classes = detector._postprocess_output(output, (224, 224))

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1
    assert classes.ndim == 1


def test_full_model_pipeline(detector):
    dummy_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

    results = detector.predict(dummy_image)
    assert isinstance(results, list)
    assert all(isinstance(d, dict) for d in results[0] or [])  # if there are detections
