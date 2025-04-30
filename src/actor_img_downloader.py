from src.face_cropper import FaceCropper
from src.face_detector import FaceDetector
from src.model import FaceFeatureExtractor


class ActorDatabase:
    def __init__(
        self,
        tmdb_api_key: str,
        actor_names: list[str],
        extractor: FaceFeatureExtractor,
        detector: FaceDetector,
        cropper: FaceCropper,
    ):
        pass
