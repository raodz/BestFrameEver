import pytest
import torch

from src.model import FeatureExtractor


def test_feature_extractor():
    model = FeatureExtractor()
    model.eval()
    x = torch.randn(1, 3, 448, 448)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 2048, 7, 7), f"Unexpected output shape: {output.shape}"


@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_detector_output_shape(detector, batch_size):
    dummy_input = torch.randn(batch_size, 2048, 7, 7)
    output = detector(dummy_input)
    assert output.shape == (batch_size, 7, 7, 30)


def test_detector_output_values_are_finite(detector):
    dummy_input = torch.randn(1, 2048, 7, 7)
    output = detector(dummy_input)
    assert torch.isfinite(output).all()
