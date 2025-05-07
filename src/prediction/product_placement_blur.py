from typing import Tuple

import numpy as np


class ProductPlacementBlur:
    def __init__(
        self, frame: np.ndarray, pp_bbox: Tuple[float, float, float, float] | None
    ):
        self.frame = frame
        self.pp_bbox = pp_bbox

    def blur_product_placement(self) -> np.ndarray:
        pass
