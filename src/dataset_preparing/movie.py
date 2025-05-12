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
        self._cap: cv2.VideoCapture | None = None

    @property
    def cap(self) -> cv2.VideoCapture | None:
        """
        Get the video capture object.

        Returns:
            cv2.VideoCapture: The video capture object.
        """
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.file_path)
            if not self._cap.isOpened():
                logging.error(f"Could not open video file at {self.file_path}")
                self._cap = None
            else:
                logging.info(f"Video opened successfully from {self.file_path}")
        return self._cap

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
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logging.info("Video capture released")
            self._cap = None
        else:
            logging.warning("Video capture is not initialized or already released")

    def __del__(self) -> None:
        """
        Destructor to ensure resources are released when the object is deleted.
        """
        self.release()
