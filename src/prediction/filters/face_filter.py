from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.detectors.face_detector import FaceDetector
from src.prediction.filters.base_frame_filter import BaseFrameFilter


class FaceFilter(BaseFrameFilter):
    def _init_detector(self) -> FaceDetector:
        return FaceDetector()

    def _get_filtering_conditions(
        self, dataset: FrameDataset, detector: FaceDetector
    ) -> dict[int, bool]:
        return detector.detect(dataset)

    def _filter_dataset(
        self, dataset: FrameDataset, filtering_conditions: dict[int, bool]
    ) -> FrameDataset:
        pass
