from functools import reduce

from src.dataset_preparing.frame_dataset import FrameDataset
from src.prediction.filters.base_frame_filter import BaseFrameFilter
from src.utils.utils import compose, identity


class FilterPipeline:
    def __init__(self, filters: list[type[BaseFrameFilter]]):
        self.filters = filters
        filter_functions = [f.filter for f in self.filters]
        self._composed_filter_function = reduce(compose, filter_functions, identity)

    def filter(self, dataset: FrameDataset) -> FrameDataset:
        return self._composed_filter_function(dataset)
