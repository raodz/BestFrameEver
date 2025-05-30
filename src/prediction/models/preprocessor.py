import cv2
import numpy as np
import torch

from src.constants import RGB_CHANNELS


class Preprocessor:
    def __init__(
        self,
        input_size: tuple[int, int],
        mean: list[float],
        std: list[float],
        scale_factor: float = 255.0,
    ):
        self.input_size = input_size  # (width, height)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.scale_factor = scale_factor

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[2] == RGB_CHANNELS:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        normalized = resized.astype(np.float32) / self.scale_factor
        normalized = (normalized - self.mean) / self.std

        # to tensor (C, H, W, batch=1)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor
