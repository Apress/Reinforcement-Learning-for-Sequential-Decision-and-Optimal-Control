"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4:  data sampler

"""

from typing import Dict, List, Union
import numpy as np


class Sampler:
    """ Class to log data during training process. A simple but NOT elegant implementation. """
    def __init__(self, sample_rate: float = 1):
        assert 0 <= sample_rate <= 1

        self.sample_rate = sample_rate
        self._calls = 0
        self._counter = 0
        self.data: Dict[str, Union[np.ndarray, List]] = {}
        self.episode: Dict[str, List[int]] = {}

    def sample(self, data: Dict[str, Union[np.ndarray, List]], irregular: bool = False):
        self._calls += 1
        self._counter += 1
        if self._calls * self.sample_rate >= 1:
            self._counter = 0
            self.log(self._calls, data, irregular)

    def log(self, episode: int, data: Dict[str, Union[np.ndarray, List]], irregular: bool = False):
        for k, v in data.items():
            if irregular:
                if k in self.data:
                    self.data[k].append(v)
                    self.episode[k].append(episode)
                else:
                    self.data[k] = [v]
                    self.episode[k] = [episode]
            else:
                v = v[..., np.newaxis].copy()
                if k in self.data:
                    self.data[k] = np.concatenate((self.data[k], v), axis=-1)
                    self.episode[k].append(episode)
                else:
                    self.data[k] = v
                    self.episode[k] = [episode]
