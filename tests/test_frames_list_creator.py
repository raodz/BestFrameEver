import numpy as np
import pytest

from src.dataset_preparing.frames_list_creator import FramesListCreator


def test_frames_list_creator_initialization(movie):
    flc = FramesListCreator(movie)
    assert flc.movie == movie


@pytest.mark.parametrize("enf", [1, 2, 5, 10])
def test_create_frames_list_creator_with_valid_interval(movie, flc, enf):
    frames = flc.create_frames_list(every_nth_frame=enf)

    assert len(frames) > 0
    assert all(isinstance(f, np.ndarray) for f in frames)


def test_create_frames_list_creator_invalid_interval_raises(flc):
    with pytest.raises(ValueError):
        flc.create_frames_list(every_nth_frame=0)
