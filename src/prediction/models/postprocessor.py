import torch

from src.utils.constants import N_BOX_COORDS


class Postprocessor:
    def __init__(self, num_boxes: int, num_classes: int, grid_size: int):
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size

    def __call__(self, output: torch.Tensor, img_size: tuple[int, int]):
        img_w, img_h = img_size
        device = output.device

        # N_BOX_COORDINATES + confidence_score + n_classes
        output_dim = N_BOX_COORDS + 1 + self.num_classes
        output = output.view(
            -1, self.grid_size, self.grid_size, self.num_boxes, output_dim
        )

        grid_x = torch.arange(self.grid_size, device=device).view(
            1, 1, self.grid_size, 1
        )
        grid_y = torch.arange(self.grid_size, device=device).view(
            1, self.grid_size, 1, 1
        )

        box_xy = torch.sigmoid(output[..., :2])
        box_wh = torch.exp(output[..., 2:4])
        box_conf = torch.sigmoid(output[..., 4:5])
        class_probs = torch.softmax(output[..., 5:], dim=-1)

        x = (box_xy[..., 0] + grid_x) / self.grid_size * img_w
        y = (box_xy[..., 1] + grid_y) / self.grid_size * img_h
        w = box_wh[..., 0] * img_w
        h = box_wh[..., 1] * img_h

        boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)

        class_scores, class_ids = torch.max(class_probs, dim=-1)
        scores = box_conf.squeeze(-1) * class_scores

        return boxes.view(-1, 4), scores.view(-1), class_ids.view(-1)
