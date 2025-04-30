from typing import Dict, Tuple

from src.dataset_preparing.frame_dataset import FrameDataset


class MainCharacterDetector:
    def __init__(self, dataset: FrameDataset):
        self.dataset = dataset

    def detect_main_char_per_frame(
        self,
    ) -> Dict[int, Tuple[float, float, float, float] | None]:
        pass
