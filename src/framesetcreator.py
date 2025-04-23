import logging

import numpy as np

from src.movie import Movie


class FrameSetCreator:
    """
    A utility class for extracting frames from a video.
    It does not store any state — it just creates and returns a set of frames.

    Attributes:
        movie (Movie): The Movie object used to retrieve frames.
    """

    def __init__(self, movie: Movie):
        """
        Initialize the FrameSetCreator with a movie instance.

        :param movie: A Movie object providing access to the video.
        """
        self.movie = movie

    def create_frameset(self, every_nth_frame: int) -> list[np.ndarray]:
        """
        Extract every nth frame from the already-opened video.

        :param every_nth_frame: Interval for frame extraction, e.g. 5 means every 5th frame is taken.
        :return: A list of extracted frameset (np.ndarray).
        :raises ValueError: If the interval is less than 1.
        """
        if every_nth_frame < 1:
            raise ValueError("Frame extraction interval must be at least 1")

        if self.movie.cap is None or not self.movie.cap.isOpened():
            raise RuntimeError("Video has not been read or has been released.")

        logging.info(f"Starting frame extraction: every {every_nth_frame}th frame.")
        frameset = []
        frame_index = 0
        saved_count = 0

        while True:
            frame = self.movie.get_frame()
            if frame is None:
                break

            if frame_index % every_nth_frame == 0:
                frameset.append(frame)
                saved_count += 1

            frame_index += 1

        logging.info(f"Frame extraction completed. Total saved frameset: {saved_count}")
        return frameset
