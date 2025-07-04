import os

import numpy as np
import yaml

from src.utils.paths import PROJECT_ROOT


def compose(f, g):
    return lambda x: f(g(x))


def identity(x):
    return x


def calculate_iou(bbox: tuple[float, float, float, float], frame: np.ndarray) -> float:
    pass


def parse_bbox(x: int, y: int, w: int, h: int) -> list[int]:
    return [x, y, x + w, y + h]


def load_yaml_config(relative_path: str):
    abs_path = os.path.join(PROJECT_ROOT, "configs", relative_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
