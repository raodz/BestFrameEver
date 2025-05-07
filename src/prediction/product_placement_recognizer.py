import numpy as np


class ProductPlacementRecognizer:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def recognize(self) -> tuple[float, float, float, float]:
        pass
