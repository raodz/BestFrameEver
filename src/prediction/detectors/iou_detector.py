import numpy as np

from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.detectors.base_frame_detector import BaseFrameDetector


class IoUDetector(BaseFrameDetector):
    def calculate_iou(self, bbox: tuple[float, float, float, float], frame: np.ndarray):
        pass

    def detect(self, dataset: FrameDataset) -> dict[int, bool]:
        pass
