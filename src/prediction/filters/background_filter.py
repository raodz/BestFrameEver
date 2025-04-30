from typing import Dict

from src.dataset_preparing.frame_dataset import FrameDataset


class BackgroundFilter:
    def __init__(
        self, dataset: FrameDataset, background_status_by_frame: Dict[int, bool]
    ):
        self.dataset = dataset
        self.background_status_by_frame = background_status_by_frame

    def filter_bg_valid_frames(self) -> FrameDataset:
        pass
