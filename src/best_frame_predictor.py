import torch

from src.face_detector import FaceDetector
from src.frame_dataset import FrameDataset
from src.frames_list_creator import FramesListCreator
from src.movie import Movie


class BestFramePredictor:
    def __init__(self, detector: FaceDetector, frame_dataset: FrameDataset, top_k=10):
        self.detector = detector
        self.frame_dataset = frame_dataset
        self.top_k = top_k

    def predict(self, movie: Movie) -> list[tuple[int, torch.Tensor]]:
        """
        Returns top_k best frames
        """
        pass
