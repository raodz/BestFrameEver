import os
import re

import cv2
import numpy as np

from src.utils.constants import (
    JPEG_QUALITY_VALUE,
    N_BGRA_CHANNELS,
    N_GRAYSCALE_DIMENSIONS,
    N_MAX_RSPLIT_SPLITS,
    N_SINGLE_CHANNEL,
)


def read_image_data(response, chunk_size):
    return b"".join(response.iter_content(chunk_size=chunk_size))


def normalize_channels(img):
    if (
        img.ndim == N_GRAYSCALE_DIMENSIONS
        or img.shape[N_GRAYSCALE_DIMENSIONS] == N_SINGLE_CHANNEL
    ):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[N_GRAYSCALE_DIMENSIONS] == N_BGRA_CHANNELS:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_as_jpg(img, original_path):
    filepath_jpg = original_path.rsplit(".", N_MAX_RSPLIT_SPLITS)[0] + ".jpg"
    cv2.imwrite(filepath_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_VALUE])


def prepare_actor_dir_name(actor_name):
    name = re.sub(r"[^\w\s-]", "", actor_name).strip()
    return re.sub(r"[-\s]+", "_", name)


def build_actor_directory(output_dir, actor_name):
    dir_path = os.path.join(output_dir, prepare_actor_dir_name(actor_name))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_search_variants(actor_name):
    return [
        f"{actor_name} actor portrait",
        f"{actor_name} headshot",
        f"{actor_name} face photo",
    ]


def bytes_to_cv_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    return img
