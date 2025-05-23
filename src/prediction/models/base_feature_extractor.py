from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseFeatureExtractor(nn.Module, ABC):
    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
