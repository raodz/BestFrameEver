from unittest.mock import MagicMock, patch

import pytest

from src.actors_names_api.actors_names_finding import get_actors_names
from src.exceptions.actor_exceptions import ActorsNotFoundException
from src.utils.paths import KEY_PATH
from src.utils.utils import load_yaml_config
from tests.conftest import skip_if_key, skip_if_no_key

prompt = load_yaml_config("movie_actors//prompt//default.yaml")["template"]
vertex_ai = load_yaml_config("movie_actors//vertex_ai//default.yaml")
vertex_ai["credentials_path"] = KEY_PATH


@skip_if_key
@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_valid_mocked_api(mock_model_class):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = (
        "Jennifer Lawrence, Josh Hutcherson, Liam Hemsworth"
    )
    mock_model_class.return_value = mock_model

    result = get_actors_names("The Hunger Games", 3, prompt, vertex_ai)

    assert result == ["Jennifer Lawrence", "Josh Hutcherson", "Liam Hemsworth"]


@skip_if_key
@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_invalid_mocked_api(mock_model_class):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Cannot find actors for this movie"
    mock_model_class.return_value = mock_model

    with pytest.raises(ActorsNotFoundException):
        get_actors_names("qwertyuiop", 3, prompt, vertex_ai)


@pytest.mark.integration
@skip_if_no_key
def test_get_actors_names_valid_with_api():
    result = get_actors_names("The Hunger Games", 3, prompt, vertex_ai)

    expected = ["Jennifer Lawrence", "Josh Hutcherson", "Liam Hemsworth"]

    assert all(actor in result for actor in expected)
    assert result[0] == "Jennifer Lawrence"


@pytest.mark.integration
@skip_if_no_key
def test_get_actors_names_invalid_with_api():
    with pytest.raises(ActorsNotFoundException):
        get_actors_names("qwertyuiop", 3, prompt, vertex_ai)
