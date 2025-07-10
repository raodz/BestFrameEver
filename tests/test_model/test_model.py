import torch

from src.model import FeatureExtractor


def test_feature_extractor():
    model = FeatureExtractor()
    model.eval()
    x = torch.randn(1, 3, 448, 448)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 2048, 7, 7), f"Unexpected output shape: {output.shape}"


test_feature_extractor()
