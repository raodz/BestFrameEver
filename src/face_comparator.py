import numpy as np

from src.actor_img_downloader import ActorDatabase


class FaceComparator:
    def __init__(self, actor_db: ActorDatabase, threshold: float = 0.8):
        self.actor_db = actor_db
        self.threshold = threshold

    def is_actor_present(self, face_embedding: np.ndarray) -> bool:
        pass
