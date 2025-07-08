import torch
from torch import nn
from torchvision import models

from src.prediction.models.base.base_feature_extractor import BaseFeatureExtractor


class ResNet50FE(BaseFeatureExtractor):
    def __init__(self, grid_size: int, output_feature_channels: int):
        super().__init__()
        self.grid_size = grid_size
        self._output_shape = (output_feature_channels, grid_size, grid_size)

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        del self.resnet.avgpool
        del self.resnet.fc

        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.adaptive_pool(x)
        return x
