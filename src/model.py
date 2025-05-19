from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.ops as ops
from torchvision import models


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass


class FeatureExtractor(BaseModel):
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


class DetectionHead(BaseModel):
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


class Detector(BaseModel):
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
        self.detector = DetectionHead(num_boxes, num_classes)
        self.num_boxes = num_boxes
        self.num_classes = num_classes

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

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        img_size=(448, 448),
    ):
        """
        Predict bounding boxes and class probabilities for the input image.
          Args:
                x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).
                conf_threshold (float): Confidence threshold for filtering predictions.
                iou_threshold (float): IoU threshold for non-maximum suppression.
                img_size (tuple): Size of the input image.
        Returns:
            list: List of detected objects for each image in the batch, where each object is represented as a tuple
                  (class_id, confidence_score, bounding_box).
        """
        output = self.forward(x)
        batch_results = []

        for img_idx in range(output.size(0)):
            boxes, scores, classes = self._process_output(
                output[img_idx], img_size, conf_threshold
            )

            if len(boxes) == 0:
                batch_results.append([])
                continue

            keep = self.nms(boxes, scores, classes, iou_threshold)
            batch_results.append(
                [(classes[i].item(), scores[i].item(), boxes[i].tolist()) for i in keep]
            )

        return batch_results

    def _process_output(self, output, img_size, conf_threshold):
        """
        Process the output of the detection head to extract bounding boxes, scores, and class IDs.
        Args:
            output (torch.Tensor): Output tensor from the detection head.
            img_size (tuple): Size of the input image.
            conf_threshold (float): Confidence threshold for filtering predictions.
        Returns:
            tuple: A tuple containing:
                - boxes (torch.Tensor): Bounding boxes of shape (num_boxes, 4).
                - scores (torch.Tensor): Confidence scores of shape (num_boxes,).
                - classes (torch.Tensor): Class IDs of shape (num_boxes,).
        """
        S = 7
        img_w, img_h = img_size

        output = output.view(S, S, -1)
        boxes = []
        scores = []
        classes = []

        for i in range(S):
            for j in range(S):
                cell_output = output[i, j]

                box_data = cell_output[: self.num_boxes * 5].view(self.num_boxes, 5)
                class_probs = cell_output[self.num_boxes * 5 :]

                class_score, class_id = class_probs.max(dim=0)

                for box_idx in range(self.num_boxes):
                    x, y, w, h, conf = box_data[box_idx]

                    final_score = conf * class_score
                    if final_score < conf_threshold:
                        continue

                    x_center = (j + x) * (img_w / S)
                    y_center = (i + y) * (img_h / S)
                    width = w * img_w
                    height = h * img_h

                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    boxes.append(torch.tensor([x1, y1, x2, y2]))
                    scores.append(final_score)
                    classes.append(class_id)

        if len(boxes) == 0:
            return torch.empty(0, 4), torch.empty(0), torch.empty(0)

        return torch.stack(boxes), torch.tensor(scores), torch.tensor(classes)

    def simple_nms(self, boxes, scores, classes, iou_threshold=0.5):
        """
        Perform Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
        Args:
            boxes (torch.Tensor): Bounding boxes of shape (num_boxes, 4).
            scores (torch.Tensor): Confidence scores of shape (num_boxes,).
            classes (torch.Tensor): Class IDs of shape (num_boxes,).
            iou_threshold (float): IoU threshold for filtering overlapping boxes.
        Returns:
            list: Indices of the boxes to keep after NMS.
        """
        keep = []
        idxs = scores.argsort(descending=True)

        while len(idxs) > 0:
            curr_idx = idxs[0]
            keep.append(curr_idx)

            if len(idxs) == 1:
                break

            curr_box = boxes[curr_idx]
            curr_class = classes[curr_idx]

            rest_boxes = boxes[idxs[1:]]
            ious = torch.stack([self._iou(curr_box, box) for box in rest_boxes])

            mask = (classes[idxs[1:]] != curr_class) | (ious < iou_threshold)
            idxs = idxs[1:][mask]

        return keep

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (area1 + area2 - inter_area + 1e-6)
