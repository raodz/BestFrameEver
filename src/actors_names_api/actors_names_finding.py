import os

import vertexai
from google.api_core.exceptions import DeadlineExceeded, GoogleAPIError, InvalidArgument
from vertexai.preview.generative_models import GenerativeModel

from src.exceptions.actor_exceptions import ActorsNotFoundException
from src.utils.setup_logger import setup_logger

logger = setup_logger()


def get_actors_names(
    title: str, n_actors: int, prompt_template: str, vertex_config: dict
) -> list[str]:
    os.environ[vertex_config["env_var_name"]] = vertex_config["credentials_path"]

    vertexai.init(project=vertex_config["project_id"], location=vertex_config["region"])
    model = GenerativeModel(vertex_config["model_name"])

    prompt = prompt_template.format(movie_title=title, n_actors=n_actors)
    logger.info(f"Querying actors for movie: {title}")

    vertex_exceptions = {
        InvalidArgument: "Invalid prompt or parameters.",
        DeadlineExceeded: "Request to Vertex AI timed out.",
        GoogleAPIError: "Vertex AI API error.",
    }

    try:
        response = model.generate_content(prompt)
    except tuple(vertex_exceptions) as e:
        error_msg = vertex_exceptions.get(type(e), "Vertex AI request failed.")
        logger.error(f"{error_msg} Details: {str(e)}")
        raise

    if response and response.text:
        result = response.text.strip()
        if result == "Cannot find actors for this movie":
            logger.error(f"Actors not found for movie: {title}")
            raise ActorsNotFoundException(f"No reliable cast found for movie: {title}")
        logger.info(f"Actor names: {result}")
        return [name.strip() for name in result.split(",")]
    else:
        logger.error("Empty or missing response from model")
        raise ActorsNotFoundException("Empty response received from the model.")
