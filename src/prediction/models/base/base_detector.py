from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from src.prediction.models.base.base_detection_head import BaseDetectionHead
from src.prediction.models.base.base_feature_extractor import BaseFeatureExtractor


class BaseDetector(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _init_feature_extractor(
        self, grid_size, output_feature_channels
    ) -> BaseFeatureExtractor:
        pass

    @abstractmethod
    def _init_detection_head(
        self,
        input_feature_channels: int,
        input_grid_size: int,
        hidden_size: int,
        leaky_relu_slope: float,
    ) -> BaseDetectionHead:
        pass

    @abstractmethod
    def preprocess(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(
        self,
        inputs: np.ndarray | list[np.ndarray] | torch.Tensor,
    ) -> list[list[dict]]:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected (N,C,H,W) input, got {x.shape}"
        assert x.shape[1] == 3, "Input must have 3 color channels"

        features = self.feature_extractor(x)
        assert (
            features.shape[1:] == self.feature_extractor.output_shape
        ), f"Feature shape mismatch: {features.shape} vs {self.feature_extractor.output_shape}"

        return self.detection_head(features)
