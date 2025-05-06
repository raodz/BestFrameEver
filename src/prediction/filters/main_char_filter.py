from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.detectors.main_char_detector import MainCharacterDetector
from src.prediction.filters.base_frame_filter import BaseFrameFilter


class MainCharacterFilter(BaseFrameFilter):
    def _init_detector(self) -> MainCharacterDetector:
        return MainCharacterDetector()

    def _get_filtering_conditions(
        self, dataset: FrameDataset, detector: MainCharacterDetector
    ) -> dict[int, bool]:
        return detector.detect(dataset)

    def _filter_dataset(
        self, dataset: FrameDataset, filtering_conditions: dict[int, bool]
    ) -> FrameDataset:
        pass
