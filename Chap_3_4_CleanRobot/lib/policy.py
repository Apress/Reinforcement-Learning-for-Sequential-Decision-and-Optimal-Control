"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4:  greedy policy and epsilon-greedy policy

"""

import numpy as np


def greedy_policy(Q_value):
    """
    Greedy policy.

    :param Q_value: Estimated action value function. The last axis shall be the action axis.
    :return: New policy based on Q_value
    """
    new_policy = np.zeros_like(Q_value)  # Initialize greedy policy for each grid cell
    greedy_index = _get_greedy_mesh(new_policy.shape, np.argmax(Q_value, axis=-1))  # Get indices for argmax
    new_policy[greedy_index] = 1
    return new_policy, Q_value[greedy_index]


def epsilon_greedy_policy(Q_value, epsilon):
    """
    Epsilon-greedy policy.

    :param Q_value: Estimated action value function. The last axis shall be the action axis.
    :param epsilon: Small probability of epsilon-greedy policy
    :return: New policy based on Q_value
    """
    # TODO: Fix incorrect greedy on unvisited cells?
    num_actions = Q_value.shape[-1]
    new_policy = np.ones_like(Q_value) * (epsilon / num_actions)  # Initialize greedy policy for each grid cell
    greedy_index = _get_greedy_mesh(new_policy.shape, np.argmax(Q_value, axis=-1))  # Get indices for argmax
    new_policy[greedy_index] = 1 - epsilon * (1 - 1 / num_actions)
    return new_policy, Q_value[greedy_index]


def _get_greedy_mesh(shape, argmax):
    """
    Generate meshgrid-like tuple to allow vectorized ndarray indexing.

    :param shape: State-action array shape, last axis shall be the action axis
    :param argmax: Argmax along the action axis
    :return: ndarray index
    """
    return np.ix_(*[np.arange(axis) for axis in shape[0:-1]]) + (argmax,)  # Meshgrid-like indexing
