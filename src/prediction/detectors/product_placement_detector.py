from typing import Tuple

import numpy as np


class ProductPlacementDetector:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def detect_product_placement(self) -> Tuple[float, float, float, float] | None:
        pass
