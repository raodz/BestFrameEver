import numpy as np

from src.dataset_preparing.frame_dataset import FrameDataset


class FinalFrameIndicator:
    def __init__(self, dataset: FrameDataset):
        self.dataset = dataset

    def choose_final_frame(self) -> np.ndarray:
        pass
