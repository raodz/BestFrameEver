import os
import shutil

import cv2

from src.actor_img_downloader import ActorDatabase
from src.best_frame_predictor import BestFramePredictor
from src.face_comparator import FaceComparator
from src.face_cropper import FaceCropper
from src.face_detector import FaceDetector
from src.face_filter import ActorFaceFilter
from src.frame_dataset import FrameDataset
from src.frames_list_creator import FramesListCreator
from src.model import FaceFeatureExtractor
from src.movie import Movie
from src.ui import FrameSelectorUI


def main():
    proposed_img_dir_path = "../proposed_frames"
    chosen_img_dir_path = "../chosen_frames"

    os.makedirs(proposed_img_dir_path, exist_ok=True)
    os.makedirs(chosen_img_dir_path, exist_ok=True)

    movie = Movie(
        name="Margin Call",
        file_path=r"C:\Users\user\Projekty\BestFrameData\Margin Call.mp4",
        actors=["Kevin Spacey", "Paul Bettany", "Jeremy Irons"],
    )

    flc = FramesListCreator(movie)
    frames = flc.create_frames_list(every_nth_frame=50)

    dataset = FrameDataset(frames)

    face_detector = FaceDetector()
    feature_extractor = FaceFeatureExtractor()
    cropper = FaceCropper()

    actor_db = ActorDatabase(
        tmdb_api_key="",
        actor_names=movie.actors,
        extractor=feature_extractor,
        detector=face_detector,
        cropper=cropper,
    )

    face_comparator = FaceComparator(actor_db)

    predictor = BestFramePredictor(
        detector=face_detector, frame_dataset=dataset, top_k=10
    )

    best_frames = predictor.predict(movie)

    actor_filter = ActorFaceFilter(
        face_detector=face_detector,
        cropper=cropper,
        extractor=feature_extractor,
        comparator=face_comparator,
    )
    final_frames = actor_filter.filter(frames=best_frames)

    for idx, frame in final_frames:
        cv2.imwrite(f"{proposed_img_dir_path}/{idx}.jpg", frame)

    ui = FrameSelectorUI(proposed_img_dir_path)
    selected_frames_paths = ui.display_frames()

    for path in selected_frames_paths:
        shutil.copy(path, chosen_img_dir_path)

    shutil.rmtree(proposed_img_dir_path)
