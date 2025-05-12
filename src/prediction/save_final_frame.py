import os

import cv2
import numpy as np

import config


def save_final_frame(frame: np.ndarray):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{config.movie_title}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, frame)
