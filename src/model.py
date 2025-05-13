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

        self.resnet = models.resnet50(pretrained=True)

        del self.resnet.avgpool
        del self.resnet.fc

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


class Detector(nn.Module):
    """
    YOLO detection head using fully connected layers.

    This head takes the extracted feature map and outputs predictions for bounding boxes
    and class probabilities for each grid cell.
    """

    def __init__(self, num_boxes: int = 2, num_classes: int = 20):
        """
        Initialize the YOLOv1Detector.

        Args:
            num_boxes (int): Number of bounding boxes predicted per grid cell.
            num_classes (int): Number of object classes.
        """
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Flatten the feature map, then apply two fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * num_boxes + num_classes))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the YOLO detection head.

        Args:
            x (torch.Tensor): Feature map tensor of shape (batch_size, 2048, 7, 7).

        Returns:
            torch.Tensor: Prediction tensor of shape (batch_size, 7, 7, num_boxes*5 + num_classes).
        """
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 7, 7, 5 * self.num_boxes + self.num_classes)


class YOLO(nn.Module):
    """
    Complete YOLO model combining the feature extractor and detection head.

    This model takes an input image, extracts features using a backbone network,
    and predicts bounding boxes and class probabilities for each grid cell.
    """

    def __init__(self, num_boxes: int = 2, num_classes: int = 20):
        """
        Initialize the YOLO model.

        Args:
            num_boxes (int): Number of bounding boxes predicted per grid cell.
            num_classes (int): Number of object classes.
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.detector = Detector(num_boxes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the YOLO model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Prediction tensor of shape (batch_size, 7, 7, num_boxes*5 + num_classes).
        """
        features = self.feature_extractor(x)
        return self.detector(features)
