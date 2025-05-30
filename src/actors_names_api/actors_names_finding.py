import os

import vertexai
from vertexai.preview.generative_models import GenerativeModel

from logging_management import setup_logger

logger = setup_logger()


def get_actors_names(
    title: str, prompt_template: str, vertex_config: dict
) -> list[str] | str:
    os.environ[vertex_config["env_var_name"]] = vertex_config["credentials_path"]

    vertexai.init(project=vertex_config["project_id"], location=vertex_config["region"])
    model = GenerativeModel(vertex_config["model_name"])

    prompt = prompt_template.format(movie_title=title)
    logger.info(f"Querying actors for movie: {title}")

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        logger.error(f"Error getting movie actors: {str(e)}")
        return f"Error: {str(e)}"

    if response and response.text:
        result = response.text.strip()
        logger.info(f"Actor names: {result}")
        return [name.strip() for name in result.split(",")]
    else:
        logger.error("No response from model")
        return "Error: No response from model"
