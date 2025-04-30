from typing import Dict

from src.dataset_preparing.frame_dataset import FrameDataset


class IoUFilter:
    def __init__(self, dataset: FrameDataset, iou_per_frame: Dict[int, float]):
        self.dataset = dataset
        self.iou_per_frame = iou_per_frame

    def filter_frames_by_iou(self) -> FrameDataset:
        pass
