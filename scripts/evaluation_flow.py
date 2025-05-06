from src.evaluation.evaluation_gui import EvaluationGUI
from src.evaluation.image_folder_manager import ImageFolderManager


def run_evaluation_flow(
    saved_images_dir: str = "saved_images",
    chosen_images_dir: str = "chosen_images",
):
    folder_manager = ImageFolderManager(saved_images_dir, chosen_images_dir)
    folder_manager.open_folder()

    def on_close():
        folder_manager.print_chosen_images_metadata()

    gui = EvaluationGUI(on_close_callback=on_close)
    gui.run()


if __name__ == "__main__":
    chosen_images_dir = "sample_folder/chosen_images"
    saved_images_dir = "sample_folder/saved_images"
    run_evaluation_flow(saved_images_dir, chosen_images_dir)
