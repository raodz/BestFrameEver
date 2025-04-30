import config
from src.dataset_preparing.frame_dataset import FrameDataset
from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.prediction.calculators.face_counter import FaceCounter
from src.prediction.calculators.iou_calculator import IoUCalculator
from src.prediction.detectors.background_detector import BackgroundDetector
from src.prediction.detectors.face_detector import FaceDetector
from src.prediction.detectors.main_char_detector import MainCharacterDetector
from src.prediction.detectors.product_placement_detector import ProductPlacementDetector
from src.prediction.detectors.sharpness_detector import SharpnessDetector
from src.prediction.filters.background_filter import BackgroundFilter
from src.prediction.filters.face_filter import FaceFilter
from src.prediction.filters.iou_filter import IoUFilter
from src.prediction.filters.main_char_filter import MainCharacterFilter
from src.prediction.filters.sharpness_filter import SharpnessFilter
from src.prediction.final_frame_indicator import FinalFrameIndicator
from src.prediction.product_placement_blur import ProductPlacementBlur
from src.prediction.save_final_frame import save_final_frame


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

    # Main character IoU filtering

    iou_per_frame = IoUCalculator(
        main_char_filtered_dataset, main_char_bbox_per_frame
    ).calc_iou_per_frame()
    iou_filtered_dataset = IoUFilter(
        main_char_filtered_dataset, iou_per_frame
    ).filter_frames_by_iou()

    # Sharpen frames filtering

    sharp_status_by_frame = SharpnessDetector(
        iou_filtered_dataset
    ).detect_sharp_frames()
    sharp_filtered_dataset = SharpnessFilter(
        iou_filtered_dataset, sharp_status_by_frame
    ).filter_sharp_frames()

    # Background filtering

    background_status_by_frame = BackgroundDetector(
        sharp_filtered_dataset
    ).detect_bg_valid_frames()
    background_filtered_dataset = BackgroundFilter(
        sharp_filtered_dataset, background_status_by_frame
    ).filter_bg_valid_frames()

    # Final frame

    final_frame = FinalFrameIndicator(background_filtered_dataset).choose_final_frame()

    pp_bbox = ProductPlacementDetector(final_frame).detect_product_placement()
    final_frame_no_pp = ProductPlacementBlur(
        final_frame, pp_bbox
    ).blur_product_placement()

    save_final_frame(final_frame_no_pp)


if __name__ == "__main__":
    main()
