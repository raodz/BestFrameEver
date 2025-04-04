import logging
import os
import cv2
from typing import List
from src.Movie import Movie


class FrameSet:
    """
    A lightweight class representing a set of frames extracted from a video.
    Responsible only for frame management and access — not analysis.

    Attributes:
        movie (Movie): The Movie object used to retrieve frames.
        frames (List): The list of frames (as np.ndarrays) extracted from the video.
    """

    def __init__(self, movie: Movie):
        """
        Initialize the FrameSet with a Movie instance.

        :param movie: A Movie object providing access to the video.
        """
        self.movie = movie
        self.frames: List = []

    def get_all_frames(self, frequency: int):
        """
        Extract frames from the video with a given frequency.

        :param frequency: Every nth frame will be included.
        :raises ValueError: If frequency is less than 1.
        """
        if frequency < 1:
            raise ValueError("Frequency must be at least 1")

        logging.info(f"Starting frame extraction with frequency={frequency}")
        self.frames = []
        frame_index = 0
        saved_count = 0

        while True:
            frame = self.movie.get_frame()
            if frame is None:
                break

            if frame_index % frequency == 0:
                self.frames.append(frame)
                saved_count += 1

            frame_index += 1

        logging.info(f"Frame extraction completed. Total saved frames: {saved_count}")

    def __len__(self):
        """Return the number of frames in the set."""
        return len(self.frames)

    def __getitem__(self, index: int):
        """Allow frame access via indexing."""
        return self.frames[index]
