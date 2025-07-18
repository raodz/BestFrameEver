import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ACTORS_IMAGES_PATH = "data//actors_images"
IMAGE_VALIDATION_CONFIG_ABS_PATH = os.path.join(
    PROJECT_ROOT, "configs", "img_scrapping", "image_validation.yaml"
)

PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

MOVIE_ACTORS_REL_PATH = os.path.join("configs", "movie_actors")

KEY_PATH = os.path.join(PARENT, "bestframeever-key.json")
