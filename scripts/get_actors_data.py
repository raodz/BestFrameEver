import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.actors_names_api.actors_names_finding import get_actors_names
from src.utils.paths import MOVIE_ACTORS_REL_PATH

movie_actors_project_rel_path = os.path.join("..", MOVIE_ACTORS_REL_PATH)


@hydra.main(
    version_base=None, config_path=movie_actors_project_rel_path, config_name="default"
)
def main(cfg: DictConfig):
    actor_names = get_actors_names(
        title=cfg.movie.title,
        prompt_template=cfg.prompt.template,
        vertex_config={
            "project_id": cfg.vertex_ai.project_id,
            "region": cfg.vertex_ai.region,
            "model_name": cfg.vertex_ai.model_name,
            "credentials_path": cfg.vertex_ai.credentials_path,
            "env_var_name": cfg.vertex_ai.env_var_name,
        },
    )

    print("Main actors in this movie:", actor_names)


if __name__ == "__main__":
    main()
