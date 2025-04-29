from typing import Dict, List, Tuple

from src.frame_dataset import FrameDataset


class FaceDetector:
    def __init__(self, dataset: FrameDataset):
        self.dataset = dataset

    def detect_faces_per_frame(
        self,
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        pass
