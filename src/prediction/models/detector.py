import numpy as np
import torch

from src.prediction.models.base.base_detection_head import BaseDetectionHead
from src.prediction.models.base.base_detector import BaseDetector
from src.prediction.models.base.base_feature_extractor import BaseFeatureExtractor
from src.prediction.models.postprocessor import Postprocessor
from src.prediction.models.preprocessor import Preprocessor
from src.prediction.models.resnet50fe import ResNet50FE
from src.prediction.models.yolo_v1 import YoloV1
from src.utils.utils import nms, select_device


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
            grid_size=self.grid_size,
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
        grid_size: int,
        hidden_size: int,
        leaky_relu_slope: float,
    ) -> BaseDetectionHead:
        return YoloV1(
            input_feature_channels=input_feature_channels,
            grid_size=grid_size,
            num_boxes=self.num_boxes,
            num_classes=self.num_classes,
            hidden_size=hidden_size,
            leaky_relu_negative_slope=leaky_relu_slope,
        )

    def preprocess(self, x):
        return self.preprocessor(x).to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        inputs: list[np.ndarray] | list[torch.Tensor],
    ) -> list[list[dict]]:
        if not isinstance(inputs, list):
            raise TypeError("Input must be a list of numpy arrays or tensors")

        # Preprocessing
        processed = list(map(self.preprocess, inputs))
        batch = torch.cat(processed, dim=0)

        # Model inference
        outputs = self(batch)

        results = []
        for i, (output, img) in enumerate(zip(outputs, inputs)):
            # Pobieranie rozmiaru obrazu
            if isinstance(img, np.ndarray):  # (opencv format HWC)
                h, w = img.shape[:2]
                img_size = (w, h)
            else:  # (torch format CHW)
                img_size = (img.shape[-1], img.shape[-2])

            # Postprocessing
            boxes, scores, classes = self.postprocessor(output, img_size)

            # Confidence filter
            keep_idx = torch.where(scores > self.conf_threshold)[0]
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            classes = classes[keep_idx]

            # Non-Max Suppression
            if boxes.numel() > 0:
                nms_keep = nms(boxes, scores, self.iou_threshold)
                boxes = boxes[nms_keep]
                scores = scores[nms_keep]
                classes = classes[nms_keep]

            # Results format
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

        return results
