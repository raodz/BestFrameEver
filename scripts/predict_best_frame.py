import config
from src.dataset_preparing.frame_dataset import FrameDataset
from src.dataset_preparing.frames_list_creator import FramesListCreator
from src.dataset_preparing.movie import Movie
from src.prediction.filters.face_filter import FaceFilter
from src.prediction.filters.filter_pipeline import FilterPipeline
from src.prediction.filters.iou_filter import IoUFilter
from src.prediction.filters.main_char_filter import MainCharacterFilter
from src.prediction.filters.sharpness_filter import SharpnessFilter
from src.prediction.final_frame_indicator import FinalFrameIndicator
from src.prediction.product_placement_blur import ProductPlacementBlur
from src.prediction.product_placement_recognizer import ProductPlacementRecognizer
from src.prediction.save_final_frame import save_final_frame


def main():
    # Prepare dataset

    movie_path = config.movie_path
    movie = Movie(config.movie_title, movie_path, actors=config.actors_list)
    flc = FramesListCreator(movie)
    frames_list = flc.create_frames_list(every_nth_frame=config.filter_every_nth_frame)
    dataset = FrameDataset(frames_list, transform=config.frameset_transforms)

    # Filtering

    filters = [FaceFilter, MainCharacterFilter, IoUFilter, SharpnessFilter]
    filtering_pipeline = FilterPipeline(filters)
    filtered_dataset = filtering_pipeline.filter(dataset)

    # Final frame

    final_frame = FinalFrameIndicator(filtered_dataset).choose_final_frame()

    pp_bbox = ProductPlacementRecognizer(final_frame).recognize()
    final_frame_no_pp = ProductPlacementBlur(
        final_frame, pp_bbox
    ).blur_product_placement()

    save_final_frame(final_frame_no_pp)


if __name__ == "__main__":
    main()
