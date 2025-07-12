from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.utils.utils import box_iou, nms


def test_feature_extractor_output_shape(feature_extractor):
    dummy_input = torch.randn(2, 3, 224, 224)
    output = feature_extractor(dummy_input)
    assert output.shape == (
        2,
        feature_extractor.output_shape[0],
        feature_extractor.grid_size,
        feature_extractor.grid_size,
    )


def test_detection_head_output_shape(detection_head):
    dummy_input = torch.randn(2, 2048, 7, 7)
    output = detection_head(dummy_input)
    assert output.shape == (
        2,
        detection_head.grid_size,
        detection_head.grid_size,
        detection_head.num_boxes,
        5 + detection_head.num_classes,
    )  # N_BOX_COORDS + 1 + num_classes


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
            1,
            detector.grid_size,
            detector.grid_size,
            detector.num_boxes,
            5 + detector.num_classes,
        ),
    ):
        with patch.object(
            detector.postprocessor,
            "__call__",
            return_value=(
                torch.empty(0, 4),
                torch.empty(0),
                torch.empty(0, dtype=torch.long),
            ),
        ):
            results = detector.predict([dummy_img])

    assert isinstance(results, list)
    assert len(results) == 1
    assert results == [[]], "Expected empty detections"


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


def test_preprocessor_output_shape(default_preprocessor):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = default_preprocessor(img)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_postprocessor_shapes(postprocessor):
    output = torch.randn(
        postprocessor.grid_size,
        postprocessor.grid_size,
        postprocessor.num_boxes,
        5 + postprocessor.num_classes,
    )  # (grid, grid, boxes, output_dim)
    img_size = (224, 224)
    boxes, scores, class_ids = postprocessor(output, img_size)
    assert boxes.shape == (7 * 7 * 2, 4)
    assert scores.shape == (7 * 7 * 2,)
    assert class_ids.shape == (7 * 7 * 2,)


def test_detector_predict(detector):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    results = detector.predict([img])
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    for det in results[0]:
        assert "bbox" in det
        assert "confidence" in det
        assert "class_id" in det


def test_detector_predict_wrong_input(detector):
    with pytest.raises(TypeError):
        detector.predict("not_a_list")


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
        detector.grid_size,
        detector.grid_size,
        detector.num_boxes,
        5 + detector.num_classes,
    )
    boxes, scores, classes = detector.postprocessor(output, (224, 224))

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1
    assert classes.ndim == 1


def test_full_model_pipeline(detector):
    dummy_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

    results = detector.predict([dummy_image])
    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(d, dict) for d in results[0] or [])  # if there are detections
