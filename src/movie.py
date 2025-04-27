import logging

import cv2
import numpy as np


class Movie:
    """
    A class to represent a movie and manage video using OpenCV.
    """

    def __init__(
        self, name: str, file_path: str, actors: list[str] | None = None
    ) -> None:
        """
        Initialize the Movie object.

        :param name: The name of the movie.
        :param actors: A list of actors in the movie.
        """
        self.name: str = name
        self.file_path: str = file_path
        self.actors: list[str] = actors if actors is not None else []
        self.cap: cv2.VideoCapture | None = None

    def read_video(self) -> "Movie":
        """
        Open a video file and initialize the capture object.

        Returns:
            movie: The Movie instance with initialized video capture.
        """
        self.cap = cv2.VideoCapture(self.file_path)

        if not self.cap.isOpened():
            logging.error(f"Could not open video file at {self.file_path}")
        else:
            logging.info(f"Video opened successfully from {self.file_path}")

        return self

    def get_frame(self) -> np.ndarray | None:
        """
        Retrieve the next frame from the video.

        :return: The next frame as a numpy array, or None if no more frames are available or an error occurs.
        """
        if self.cap is None or not self.cap.isOpened():
            logging.error("Video capture is not initialized or already released")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logging.warning("No more frames")
            return None

        return frame

    def release(self) -> None:
        """
        Release the resources associated with the video capture object.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logging.info("Video capture released")
        else:
            logging.warning("Video capture is not initialized or already released")

    def __del__(self) -> None:
        """
        Destructor to ensure resources are released when the object is deleted.
        """
        self.release()
