from typing import Dict

from src.dataset_preparing.frame_dataset import FrameDataset


class SharpnessFilter:
    def __init__(self, dataset: FrameDataset, sharp_status_by_frame: Dict[int, bool]):
        self.dataset = dataset
        self.sharp_status_by_frame = sharp_status_by_frame

    def filter_sharp_frames(self) -> FrameDataset:
        pass
