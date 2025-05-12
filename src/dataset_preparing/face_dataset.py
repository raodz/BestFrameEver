import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils import parse_bbox


class FaceDataset(Dataset):
    def __init__(self, images_root: str, annotation_file: str, transform=None):
        self.images_root = images_root
        self.annotation_file = annotation_file
        self.transform = transform
        self.df = self._load_annotations()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transform:
            image = self.transform(image)

        boxes = torch.tensor(row["boxes"], dtype=torch.float32)
        target = {"boxes": boxes}
        return image, target

    def _load_annotations(self) -> pd.DataFrame:
        data = []
        with open(self.annotation_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            image_rel_path = lines[i].strip()
            i += 1
            num_boxes = int(lines[i].strip())
            i += 1

            boxes = []
            for _ in range(num_boxes):
                parts = lines[i].strip().split()
                x, y, w, h = map(int, parts[:4])
                if w > 0 and h > 0:
                    boxes.append(parse_bbox(x, y, w, h))
                i += 1

            full_path = os.path.join(self.images_root, image_rel_path)
            data.append({"image_path": full_path, "boxes": boxes})

        return pd.DataFrame(data)
