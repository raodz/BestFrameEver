from abc import ABC, abstractmethod

from src.dataset_preparing.frame_dataset import FrameDataset


class BaseFrameDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, dataset: FrameDataset) -> dict[int, any]:
        pass
