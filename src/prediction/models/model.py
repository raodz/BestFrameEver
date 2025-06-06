import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torchvision import models

from src.constants import N_BOX_COORDS
from src.prediction.models.base_detection_head import BaseDetectionHead
from src.prediction.models.base_detector import BaseDetector
from src.prediction.models.base_feature_extractor import BaseFeatureExtractor
from src.prediction.models.postprocessor import Postprocessor
from src.prediction.models.preprocessor import Preprocessor
from utils import nms, select_device


class ResNet50FE(BaseFeatureExtractor):
    def __init__(self, grid_size, output_feature_channels):
        super().__init__()
        self.grid_size = grid_size
        self._output_shape = (output_feature_channels, self.grid_size, self.grid_size)

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        del self.resnet.avgpool
        del self.resnet.fc

        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))

    @property
    def output_shape(self):
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


class YoloV1(BaseDetectionHead):
    def __init__(
        self,
        input_feature_channels: int,
        input_grid_size: int,
        num_boxes,
        num_classes,
        hidden_size,
        leaky_relu_negative_slope,
        output_scaling,
    ):
        super().__init__((input_feature_channels, input_grid_size, input_grid_size))
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.output_scaling = output_scaling

        in_features = input_feature_channels * input_grid_size * input_grid_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_negative_slope)
        self.fc2 = nn.Linear(
            self.hidden_size,
            self.output_scaling
            * self.output_scaling
            * self.num_boxes
            * (5 + self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(
            -1,
            self.output_scaling,
            self.output_scaling,
            self.num_boxes,
            5 + self.num_classes,
        )


class Detector(BaseDetector):
    def __init__(
        self,
        # Main params
        input_size: tuple[int, int],
        num_boxes: int,
        num_classes: int,
        grid_size: int,
        conf_threshold: float,
        iou_threshold: float,
        # Feature Extractor params
        feature_extractor_output_channels: int,
        # Detection Head params
        detection_head_input_feature_channels: int,
        detection_head_hidden_size: int,
        detection_head_leaky_relu_slope: float,
        # Preprocessor and Postprocessor objects
        preprocessor: Preprocessor,
        postprocessor: Postprocessor,
        # Device
        device: str,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = select_device(device)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.feature_extractor = self._init_feature_extractor(
            grid_size=self.grid_size,
            output_feature_channels=feature_extractor_output_channels,
        )
        self.detection_head = self._init_detection_head(
            input_feature_channels=detection_head_input_feature_channels,
            input_grid_size=self.grid_size,
            hidden_size=detection_head_hidden_size,
            leaky_relu_slope=detection_head_leaky_relu_slope,
        )
        self.to(self.device)

    def _init_feature_extractor(
        self, grid_size: int, output_feature_channels: int
    ) -> BaseFeatureExtractor:
        return ResNet50FE(
            grid_size=grid_size, output_feature_channels=output_feature_channels
        )

    def _init_detection_head(
        self,
        input_feature_channels: int,
        input_grid_size: int,
        hidden_size: int,
        leaky_relu_slope: float,
    ) -> BaseDetectionHead:
        return YoloV1(
            input_feature_channels=input_feature_channels,
            input_grid_size=input_grid_size,
            num_boxes=self.num_boxes,
            num_classes=self.num_classes,
            hidden_size=hidden_size,
            leaky_relu_negative_slope=leaky_relu_slope,
            output_scaling=self.grid_size,
        )

    def preprocess(self, x):
        return self.preprocessor(x).to(self.device)

    def predict(
        self,
        inputs: np.ndarray | list[np.ndarray] | torch.Tensor,
    ) -> list[list[dict]]:
        inputs_list = [inputs] if not isinstance(inputs, list) else inputs
        processed = list(map(self.preprocess, inputs_list))
        batch = torch.cat(processed, dim=0)

        with torch.no_grad():
            outputs = self.forward(batch)

        results = []
        for output, img in zip(outputs, inputs_list):
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                img_size = (w, h)
            else:
                img_size = (img.shape[-1], img.shape[-2])

            boxes, scores, classes = self.postprocessor(output, img_size)

            mask = scores >= self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            if boxes.numel() > 0:
                keep = nms(boxes, scores, self.iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                classes = classes[keep]

            results.append(
                [
                    {
                        "bbox": box.cpu().numpy().tolist(),
                        "confidence": score.item(),
                        "class_id": cls.item(),
                    }
                    for box, score, cls in zip(boxes, scores, classes)
                ]
            )

            return results if isinstance(inputs, list) else results[0]
