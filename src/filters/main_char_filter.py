from typing import Dict, Tuple

from src.frame_dataset import FrameDataset


class MainCharacterFilter:
    def __init__(
        self,
        dataset: FrameDataset,
        main_char_bbox_per_frame: Dict[int, Tuple[float, float, float, float]],
    ):
        self.dataset = dataset
        self.num_faces_per_frame = main_char_bbox_per_frame

    def filter_frames_with_main_char(self) -> FrameDataset:
        pass
