from typing import Dict, Tuple

from src.dataset_preparing.frame_dataset import FrameDataset


class IoUCalculator:
    def __init__(
        self,
        dataset: FrameDataset,
        idx_bboxes_dict: Dict[int, Tuple[float, float, float, float]],
    ):
        self.dataset = dataset
        self.idx_bboxes_dict = idx_bboxes_dict

    def calc_iou_per_frame(self) -> Dict[int, float]:
        pass
