import config
from src.calculators.face_counter import FaceCounter
from src.calculators.iou_calculator import IoUCalculator
from src.detectors.face_detector import FaceDetector
from src.detectors.main_char_detector import MainCharacterDetector
from src.filters.face_filter import FaceFilter
from src.filters.iou_filter import IoUFilter
from src.filters.main_char_filter import MainCharacterFilter
from src.frame_dataset import FrameDataset
from src.frames_list_creator import FramesListCreator
from src.movie import Movie


def main():
    # Prepare dataset

    movie_path = config.movie_path
    movie = Movie(config.movie_title, movie_path, actors=config.actors_list)
    flc = FramesListCreator(movie)
    frames_list = flc.create_frames_list(every_nth_frame=config.filter_every_nth_frame)
    dataset = FrameDataset(frames_list, transform=config.frameset_transforms)

    # Face filtering

    faces_list_per_frame = FaceDetector(dataset).detect_faces_per_frame()
    num_faces_per_frame = FaceCounter(faces_list_per_frame).count_num_faces_per_frame()
    face_filtered_dataset = FaceFilter(
        dataset, num_faces_per_frame
    ).filter_frames_by_face_count()

    # Main character filtering

    main_char_bbox_per_frame = MainCharacterDetector(
        face_filtered_dataset
    ).detect_main_char_per_frame()
    main_char_filtered_dataset = MainCharacterFilter(
        face_filtered_dataset, main_char_bbox_per_frame
    ).filter_frames_with_main_char()
    main_char_iou_per_frame = IoUCalculator(
        main_char_filtered_dataset, main_char_bbox_per_frame
    ).calc_iou_per_frame()
    main_char_iou_filtered_dataset = IoUFilter(
        main_char_filtered_dataset, main_char_iou_per_frame
    ).filter_frames_by_iou()

    # Blur filtering


if __name__ == "__main__":
    main()
