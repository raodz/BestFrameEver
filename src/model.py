import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """Feature extractor using a pre-trained ResNet-50 model.

    This class uses a pre-trained ResNet-50 model as a backbone to extract features
    from input images. The output features are then passed through an adaptive average
    pooling layer to produce a fixed-size output.
    """

    def __init__(self) -> None:
        """Initialize the FeatureExtractor.

        Initializes the base ResNet50 model, removes the two final layers (average pooling and fully connected),
        and adds adaptive average pooling to ensure consistent output dimensions.
        """

        super().__init__()

        resnet = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model for feature extraction.

        Processes input through the modified ResNet50 backbone followed by adaptive pooling to produce fixed-size
        spatial features, which can by used by YOLOv1 detection head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Feature map of shape (batch_size, 2048, 7,7).
        """

        features = self.backbone(x)
        return self.adaptive_pool(features)
