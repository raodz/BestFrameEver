import os

import numpy as np
import torch
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


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Parameters
    ----------
    box1 : torch.Tensor
        Tensor of shape (N, 4), where each row is a box defined as [x1, y1, x2, y2].
    box2 : torch.Tensor
        Tensor of shape (M, 4), where each row is a box defined as [x1, y1, x2, y2].

    Returns
    -------
    torch.Tensor
        IoU matrix of shape (N, M), where the value at [i, j] is the IoU between box1[i] and box2[j].

    Notes
    -----
    The boxes must be in [x1, y1, x2, y2] format, where
        (x1, y1) is the top-left corner,
        (x2, y2) is the bottom-right corner.
    The function handles broadcasting internally to compute pairwise IoU.
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        best = idxs[0]
        keep.append(best)
        ious = box_iou(boxes[best].unsqueeze(0), boxes[idxs[1:]])
        idxs = idxs[1:][ious[0] <= iou_threshold]

    return torch.tensor(keep)


def select_device(device_config: str = None):
    if device_config not in ["cpu", "cuda"]:
        raise ValueError("Specify correct device in the config")
    return torch.device(device_config)
