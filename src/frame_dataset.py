import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.frames_list_creator import FramesListCreator
from src.movie import Movie


class FrameDataset(Dataset):
    """
    PyTorch-compatible dataset for a list of video frames (numpy arrays).

    Attributes:
        frames (List[np.ndarray]): List of frames extracted from a video.
        transform (Callable, optional): Optional transformation to apply to each frame.
    """

    def __init__(self, frames: list[np.ndarray], transform=None):
        """
        Initialize the dataset with frames and an optional transform.

        :param frames: List of video frames (as numpy arrays).
        :param transform: A callable to apply to each frame (e.g., torchvision transforms).
        """
        self.frames = frames
        self.transform = transform
        self._len = len(self.frames)

    def __len__(self) -> int:
        """
        Return the number of frames in the dataset.
        """
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve a single frame, applying the transform if available.

        :param idx: Index of the frame to retrieve.
        :return: Transformed frame as a torch.Tensor.
        """

        frame = self.frames[idx]
        frame = torch.from_numpy(frame)  # (H, W, C) —> Tensor (H, W, C)
        frame = frame.permute(2, 0, 1).float() / 255  # (C, H, W), float, [0, 1]
        return frame


def main():
    movie = Movie("The Sample", actors=[])
    movie.read_video(os.path.join(os.path.dirname(__file__), "sample_mp4_video.mp4"))
    frame = movie.get_frame()
    # print(type(frame), frame.shape, np.unique(frame))
    flc = FramesListCreator(movie)
    frames_list = flc.create_frames_list(every_nth_frame=50)
    print(frames_list[0])
    fds = FrameDataset(frames_list)
    some_frame = fds.__getitem__(11)
    print(some_frame)


main()
