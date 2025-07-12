import torch
from torch import nn

from src.prediction.models.base.base_detection_head import BaseDetectionHead
from src.utils.constants import N_BOX_COORDS


class YoloV1(BaseDetectionHead):
    def __init__(
        self,
        input_feature_channels: int,
        grid_size: int,
        num_boxes: int,
        num_classes: int,
        hidden_size: int,
        leaky_relu_negative_slope: float,
    ):
        super().__init__((input_feature_channels, grid_size, grid_size))
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size

        in_features = input_feature_channels * grid_size * grid_size

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)

        out_features = (
            grid_size * grid_size * num_boxes * (N_BOX_COORDS + 1 + num_classes)
        )
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(
            -1,
            self.grid_size,
            self.grid_size,
            self.num_boxes,
            N_BOX_COORDS + 1 + self.num_classes,
        )
