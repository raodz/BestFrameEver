from typing import Dict, List, Tuple


class FaceCounter:
    def __init__(
        self, faces_at_frames: Dict[int, List[Tuple[float, float, float, float]]]
    ):
        self.faces_at_frames = faces_at_frames

    def count_num_faces_per_frame(self) -> Dict[int, int]:
        pass
