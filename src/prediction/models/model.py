import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from src.prediction.models.base_detection_head import BaseDetectionHead
from src.prediction.models.base_detector import BaseDetector
from src.prediction.models.base_feature_extractor import BaseFeatureExtractor
from utils import nms


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._output_shape = (2048, 7, 7)

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        del self.resnet.avgpool
        del self.resnet.fc

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

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


class DetectionHead(BaseDetectionHead):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = 20,
        num_boxes: int = 2,
    ):
        super().__init__(input_shape)
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        in_features = input_shape[0] * input_shape[1] * input_shape[2]
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 4096)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(4096, 7 * 7 * num_boxes * (5 + num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 7, 7, self.num_boxes, 5 + self.num_classes)


class Detector(BaseDetector):
    def __init__(self, num_classes: int = 20, num_boxes: int = 2, device=None):
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.to(self.device)

    def _init_feature_extractor(self) -> BaseFeatureExtractor:
        return FeatureExtractor()

    def _init_detection_head(self) -> BaseDetectionHead:
        return DetectionHead(
            input_shape=self.feature_extractor.output_shape,
            num_classes=self.num_classes,
            num_boxes=self.num_boxes,
        )

    def preprocess(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[2] == 3:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0)

        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x.to(self.device)

    def _postprocess_output(
        self, output: torch.Tensor, img_size: tuple[int, int]
    ) -> tuple:
        img_w, img_h = img_size
        device = output.device

        output = output.view(-1, 7, 7, self.num_boxes, 5 + self.num_classes)

        grid_x = torch.arange(7, device=device).view(1, 1, 7, 1)
        grid_y = torch.arange(7, device=device).view(1, 7, 1, 1)

        box_xy = torch.sigmoid(output[..., :2])
        box_wh = torch.exp(output[..., 2:4])
        box_conf = torch.sigmoid(output[..., 4:5])
        class_probs = torch.softmax(output[..., 5:], dim=-1)

        x = (box_xy[..., 0] + grid_x) / 7 * img_w
        y = (box_xy[..., 1] + grid_y) / 7 * img_h
        w = box_wh[..., 0] * img_w
        h = box_wh[..., 1] * img_h

        boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)

        class_scores, class_ids = torch.max(class_probs, dim=-1)
        scores = box_conf.squeeze(-1) * class_scores

        return boxes.view(-1, 4), scores.view(-1), class_ids.view(-1)

    def predict(
        self,
        inputs: np.ndarray | list[np.ndarray] | torch.Tensor,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
    ) -> list[list[dict]]:
        if not isinstance(inputs, list):
            inputs = [inputs]

        processed = [self.preprocess(img) for img in inputs]
        batch = torch.cat(processed, dim=0)

        with torch.no_grad():
            outputs = self.forward(batch)

        results = []
        for i, (output, img) in enumerate(zip(outputs, inputs)):
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                img_size = (w, h)
            else:
                img_size = (img.shape[-1], img.shape[-2])

            boxes, scores, classes = self._postprocess_output(output, img_size)

            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            if boxes.numel() > 0:
                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                classes = classes[keep]

            detections = []
            for box, score, cls in zip(boxes, scores, classes):
                detections.append(
                    {
                        "bbox": box.cpu().numpy().tolist(),
                        "confidence": score.item(),
                        "class_id": cls.item(),
                    }
                )

            results.append(detections)

        return results
