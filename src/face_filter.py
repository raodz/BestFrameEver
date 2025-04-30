import torch

from src.face_comparator import FaceComparator
from src.face_cropper import FaceCropper
from src.face_detector import FaceDetector
from src.model import FaceFeatureExtractor


class ActorFaceFilter:
    def __init__(
        self,
        face_detector: FaceDetector,
        cropper: FaceCropper,
        extractor: FaceFeatureExtractor,
        comparator: FaceComparator,
    ):
        pass

    def filter(
        self, frames: list[tuple[int, torch.Tensor]]
    ) -> list[tuple[int, torch.Tensor]]:
        pass
