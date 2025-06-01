import os

import hydra
from omegaconf import DictConfig

from src.actors_names_api.actors_names_finding import get_actors_names
from src.utils.paths import MOVIE_ACTORS_REL_PATH

movie_actors_project_rel_path = os.path.join("..", MOVIE_ACTORS_REL_PATH)


@hydra.main(
    version_base=None, config_path=movie_actors_project_rel_path, config_name="default"
)
def main(cfg: DictConfig):
    get_actors_names(
        title=cfg.movie.title,
        n_actors=cfg.movie.n_actors,
        prompt_template=cfg.prompt.template,
        vertex_config=dict(cfg.vertex_ai),
    )


if __name__ == "__main__":
    main()
