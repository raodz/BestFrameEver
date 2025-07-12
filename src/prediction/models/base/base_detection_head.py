from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseDetectionHead(nn.Module, ABC):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        self.input_shape = input_shape

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
