from typing import Dict

from src.dataset_preparing.frame_dataset import FrameDataset


class BackgroundDetector:
    def __init__(self, dataset: FrameDataset):
        self.dataset = dataset

    def detect_bg_valid_frames(self) -> Dict[int, bool]:
        pass
