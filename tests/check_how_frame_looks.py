######################################################
######################################################
######################################################
#  JUST CHECKING HOW DOES LOADED DATASET LOOK #
######################################################
######################################################
######################################################
import os

import numpy as np

from src.frame_dataset import FrameDataset
from src.frames_list_creator import FramesListCreator
from src.movie import Movie

movie = Movie("The Sample", actors=[])
movie.read_video(os.path.join(os.path.dirname(__file__), "sample_avi_video.avi"))
frame = movie.get_frame()
print(type(frame), frame.shape, np.unique(frame))
# flc = FramesListCreator(movie)
# frames_list = flc.create_frames_list(every_nth_frame=50)
# print(frames_list[0])
# # fds = FrameDataset(frames_list)
# some_frame = fds.__getitem__(11)
# print(some_frame)
