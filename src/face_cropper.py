import numpy as np


class FaceCropper:
    def crop(
        self, frame: np.ndarray, bboxes: list[tuple[float, float, float, float]]
    ) -> list[np.ndarray]:
        """
        Returns a list of cropped faces from the image based on the provided bounding boxes.
        """
        pass
