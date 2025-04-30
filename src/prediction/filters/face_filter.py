from typing import Dict

from src.dataset_preparing.frame_dataset import FrameDataset


class FaceFilter:
    def __init__(self, dataset: FrameDataset, num_faces_per_frame: Dict[int, int]):
        self.dataset = dataset
        self.num_faces_per_frame = num_faces_per_frame

    def filter_frames_by_face_count(self) -> FrameDataset:
        pass
