from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.detectors.base_frame_detector import BaseFrameDetector


class ProductPlacementDetector(BaseFrameDetector):
    def detect(self, dataset: FrameDataset) -> dict[int, bool]:
        pass
