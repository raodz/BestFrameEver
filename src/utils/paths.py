import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

MOVIE_ACTORS_REL_PATH = os.path.join("configs", "movie_actors")

KEY_PATH = os.path.join(PARENT, "bestframeever-key.json")
