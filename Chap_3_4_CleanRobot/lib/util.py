"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4:  utility functions

"""

from typing import Tuple, Sequence
import numpy as np
import numba
import datetime


@numba.njit()
def is_valid_index(index: Tuple[int, ...], shape: Tuple[int, ...]) -> bool:
    """
    Check if index is within the shape bound.

    :param index: Non-negative array index
    :param shape: Array shape
    :return: True for index within shape, False for non-match or out-of-shape index
    """
    if len(index) != len(shape):
        return False
    for i, s in zip(index, shape):
        if not 0 <= i < s:
            return False
    return True

def elementwise_add(*args: Sequence[float], cast_as_original: bool = True):
    """
    Add List, Tuple or other sequences by element. (Currently unused)
    :param args: List of arguments with same length
    :param cast_as_original: Try to case the result as the same type of args[0]
    :return: Elementwise sum. Of type `type(arg[0])` if cast_as_original is True, otherwise List
    """
    if len(args) == 0:
        return []
    res = [sum([arg[i] for arg in args]) for i in range(len(args[0]))]
    if cast_as_original:
        raw_type = type(args[0])
        return raw_type(res)
    else:
        return res


def repeat(count):
    """
    Decorator to repeat a function certain times, and return as a list.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            return [func(*args, **kwargs) for _ in range(count)]

        return wrapper

    return decorator


class RingBuffer:
    """
    A 1D fixed-length ring buffer using np array
    """

    def __init__(self, length, dtype=np.int32):
        """
        :param length: Ring buffer length
        :param dtype: Data type, default to `np.int32`
        """
        self._data = np.zeros(length, dtype=dtype)
        self._index = 0
        self._length = length
        self._current_length = 0

    def append(self, x):
        """
        Add new element x to buffer. Overwrite old element if filled.

        :param x: Element to add
        """
        self._data[self._index] = x
        self._index = (self._index + 1) % self._length
        if self._current_length < self._length:
            self._current_length += 1

    def get(self):
        """
        Returns data in the ring buffer as if it's a queue. NOT recommended to call when not filled
        """
        idx = (self._index + np.arange(self._length)) % self._length
        return self._data[idx]

    def head(self):
        """
        Returns the first element in the buffer.
        """
        return self._data[self._index]

    def filled(self):
        """
        Returns if the buffer is filled.
        """
        return self._current_length == self._length

    def reset(self):
        """
        Reset to original state.
        """
        self._index = 0
        self._current_length = 0

    def __len__(self):
        return self._current_length


# ================ dump object for future plotting ================
def dump(filename, data):
    """
    :param filename: filename to write to
    :param data: object to dump
    """
    import pickle
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


# ================ load object from file system ================
def load(filename):
    """
    :param filename: filename
    :return: object loaded
    """
    import pickle
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


# ================ save agent ================
def save_agent(agent, filename='', dir_name='results/'):
    filename = dir_name + filename + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.pkl'
    dump(filename, agent)
    print('save agent in %s' % dir_name)


def load_agent(filename, dir_name='results/'):
    filename = dir_name + filename
    load(filename)
    print('load agent %s in %s' % (filename , dir_name))
