from unittest.mock import MagicMock, patch

import pytest

from src.actors_names_api.actors_names_finding import get_actors_names
from src.exceptions.actor_exceptions import ActorsNotFoundError


@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_valid(mock_model_class, dummy_vertex_config, prompt_template):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = (
        "Jennifer Lawrence, Josh Hutcherson, Liam Hemsworth"
    )
    mock_model_class.return_value = mock_model

    result = get_actors_names(
        "The Hunger Games", 3, prompt_template, dummy_vertex_config
    )

    assert result == ["Jennifer Lawrence", "Josh Hutcherson", "Liam Hemsworth"]


@patch("src.actors_names_api.actors_names_finding.GenerativeModel")
def test_get_actors_names_not_found(
    mock_model_class, dummy_vertex_config, prompt_template
):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Cannot find actors for this movie"
    mock_model_class.return_value = mock_model

    with pytest.raises(ActorsNotFoundError):
        get_actors_names("qwertyuiop", 3, prompt_template, dummy_vertex_config)
