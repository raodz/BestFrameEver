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


def test_detector_output_shape(detector, sample_input):
    backbone = FeatureExtractor()
    with torch.no_grad():
        features = backbone(sample_input)

    output = detector(features)
    batch_size = sample_input.size(0)
    assert output.shape == (batch_size, 7, 7, 30)  # 2*5 + 20 = 30


def test_full_yolo_model(sample_input):
    model = Detector()
    output = model(sample_input)
    batch_size = sample_input.size(0)
    assert output.shape == (batch_size, 7, 7, 30)


def test_detector_output_values_are_finite(detector):
    dummy_input = torch.randn(1, 2048, 7, 7)
    output = detector(dummy_input)
    assert torch.isfinite(output).all()
