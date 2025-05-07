from abc import ABC, abstractmethod

from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.detectors.base_frame_detector import BaseFrameDetector


class BaseFrameFilter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _init_detector(self) -> BaseFrameDetector:
        pass

    @abstractmethod
    def _get_filtering_conditions(
        self, dataset: FrameDataset, detector: BaseFrameDetector
    ) -> dict[int, any]:
        pass

    @abstractmethod
    def _filter_dataset(
        self, dataset: FrameDataset, filtering_conditions: dict[int, any]
    ) -> FrameDataset:
        pass

    def filter(self, dataset: FrameDataset) -> FrameDataset:
        detector = self._init_detector()
        filtering_conditions = self._get_filtering_conditions(dataset, detector)
        filtered_dataset = self._filter_dataset(dataset, filtering_conditions)
        return filtered_dataset
