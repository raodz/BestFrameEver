import pytest


@pytest.fixture(params=["sample_avi_video.avi", "sample_mp4_video.mp4"])
def sample_video_path(request):
    return request.param
