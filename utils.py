import numpy as np


def compose(f, g):
    return lambda x: f(g(x))


def identity(x):
    return x


def calculate_iou(bbox: tuple[float, float, float, float], frame: np.ndarray) -> float:
    pass


def parse_bbox(x: int, y: int, w: int, h: int) -> list[int]:
    return [x, y, x + w, y + h]
