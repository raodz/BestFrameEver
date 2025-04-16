import logging
import numpy as np
import cv2


class Movie:
    """
    A class to represent a movie and manage video using OpenCV.
    """

    def __init__(self, name: str, actors: list[str] | None = None) -> None:
        """
        Initialize the Movie object.

        :param name: The name of the movie.
        :param actors: A list of actors in the movie.
        """
        self.name: str = name
        self.actors: list[str] = actors if actors is not None else []
        self.cap: cv2.VideoCapture | None = None

    def read_video(self, file_path: str) -> "Movie":
        """
    Open a video file and initialize the capture object.

    Args:
        file_path (str): The path to the video file.

    Returns:
        Movie: The Movie instance with initialized video capture.
    """
        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            logging.error(f"Could not open video file at {file_path}")
        else:
            logging.info(f"Video opened successfully from {file_path}")

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
        logging.info(f"Movie object '{self.name}' is being deleted")
