import numpy as np
import pytest

from src.framesetcreator import FrameSetCreator


def test_frameset_initialization(movie):
    fsc = FrameSetCreator(movie)
    assert fsc.movie == movie


@pytest.mark.parametrize("every_nth_frame", [1, 2, 5, 10])
def test_create_frameset_with_valid_interval(movie, every_nth_frame):
    fsc = FrameSetCreator(movie)
    frames = fsc.create_frameset(every_nth_frame=every_nth_frame)

    assert len(frames) > 0
    assert all(isinstance(f, np.ndarray) for f in frames)


def test_create_frameset_invalid_interval_raises(frame_set_creator):
    with pytest.raises(ValueError):
        frame_set_creator.create_frameset(every_nth_frame=0)
