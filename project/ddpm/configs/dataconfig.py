from torch.utils.data import Subset, Dataset
from dataclasses import dataclass
import torch
from typing import List
from functools import cached_property


@dataclass
class DataConfig:

    dataset: Subset | Dataset

    @property
    def image_shape(self):
        return self.dataset[0][0].shape

    @property
    def image_size(self):
        assert self.image_shape[1] == self.image_shape[2], \
            "image must be a square"
        return self.image_shape[1]

    @property
    def image_channels(self):
        return self.image_shape[0]

    @property
    def image_value_range(self):
        min_, max_ = 9999, 0
        for img, _ in self.dataset:
            min_ = min(torch.min(img).item(), min_)
            max_ = max(torch.max(img).item(), max_)
        return min_, max_

    @cached_property
    def image_mean(self) -> List[float]:
        '''
        Compute mean for each channel
        '''
        channel_means = []
        for chan in range(self.image_channels):
            channel_total = torch.zeros((1, self.image_size, self.image_size))
            for img, _ in self.dataset:
                channel_total += img[chan]
            channel_means.append(
                channel_total.sum().item() /
                (len(self.dataset) * self.image_size ** 2)
            )
        return channel_means

    @cached_property
    def image_std(self):
        '''
        Compute std for each channel
        '''
        per_channel_mean = self.image_mean
        channel_stds = []
        for chan in range(self.image_channels):
            squared_error = torch.zeros((1, self.image_size, self.image_size))
            for img, _ in self.dataset:
                squared_error += (img[chan] - per_channel_mean[chan]) ** 2
            channel_stds.append((
                squared_error.sum().item() /
                (len(self.dataset) * self.image_size ** 2)) ** 0.5
            )
        return channel_stds
