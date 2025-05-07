class FaceCounter:
    def __init__(
        self, faces_at_frames: dict[int, list[tuple[float, float, float, float]]]
    ):
        self.faces_at_frames = faces_at_frames

    def count_num_faces_per_frame(self) -> dict[int, int]:
        pass
