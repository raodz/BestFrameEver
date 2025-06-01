import os
from unittest.mock import MagicMock, patch

import pytest

from src.actors_names_api.actors_names_finding import get_actors_names
from src.exceptions.actor_exceptions import ActorsNotFoundError
from src.utils.paths import KEY_PATH


@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_valid_no_api(
    mock_model_class, vertex_config, prompt_template
):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = (
        "Jennifer Lawrence, Josh Hutcherson, Liam Hemsworth"
    )
    mock_model_class.return_value = mock_model

    result = get_actors_names("The Hunger Games", 3, prompt_template, vertex_config)

    assert result == ["Jennifer Lawrence", "Josh Hutcherson", "Liam Hemsworth"]


@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_invalid_no_api(
    mock_model_class, vertex_config, prompt_template
):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Cannot find actors for this movie"
    mock_model_class.return_value = mock_model

    with pytest.raises(ActorsNotFoundError):
        get_actors_names("qwertyuiop", 3, prompt_template, vertex_config)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(KEY_PATH),
    reason="No local Vertex API key file found — skipping live test",
)
def test_get_actors_names_valid_with_api(prompt_template, vertex_config):
    result = get_actors_names("The Hunger Games", 3, prompt_template, vertex_config)

    expected = ["Jennifer Lawrence", "Josh Hutcherson", "Liam Hemsworth"]

    assert all(actor in result for actor in expected)
    assert result[0] == "Jennifer Lawrence"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(KEY_PATH),
    reason="No local Vertex API key file found — skipping live test",
)
def test_get_actors_names_invalid_with_api(prompt_template, vertex_config):
    with pytest.raises(ActorsNotFoundError):
        get_actors_names("qwertyuiop", 3, prompt_template, vertex_config)
