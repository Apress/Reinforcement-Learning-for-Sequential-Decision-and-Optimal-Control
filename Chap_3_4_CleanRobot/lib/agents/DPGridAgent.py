"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

from math import inf

import numpy as np
from tqdm import trange

from . import BaseGridAgent
from .. import greedy_policy


class DPGridAgent(BaseGridAgent):
    def __init__(self, *args, **kwargs):
        super(DPGridAgent, self).__init__(*args, **kwargs)
        self.S_value = np.zeros(self.grid_size)

    def train(self):
        """
        Dynamic programming PEV-PIM loop.
        """
        # some preparations
        _x, _y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing='ij')
        x = np.tile(_x[..., np.newaxis], (1, 1, len(self.actions)))
        x += np.array([1, 0, -1, 0]).reshape((1, 1, -1))
        x[x < 0] = 0
        x[x >= self.grid_size[0]] = self.grid_size[0] - 1

        y = np.tile(_y[..., np.newaxis], (1, 1, len(self.actions)))
        y += np.array([0, 1, 0, -1]).reshape((1, 1, -1))
        y[y < 0] = 0
        y[y >= self.grid_size[1]] = self.grid_size[1] - 1

        rewards = np.zeros(self.position_action_shape)
        rewards[:] = self.params.reward[0]
        rewards[np.logical_and(x == self.target_position[0], y == self.target_position[1])] = self.params.reward[1]

        agent_dynamics = self.agent_dynamics

        # iteration
        V_old = self.S_value
        policy = np.zeros(self.position_action_shape)
        k = 0
        error = inf
        
        iterator = trange(self.params.DP_max_iteration, desc=self.params.name, mininterval=1e-3)  # Work around for too fast loop
        for k in iterator:
            if error <= self.params.DP_error_threshold:
                iterator.close()
                print("Early stop due to error threshold")
                break
            Q_value = (agent_dynamics @ (rewards + self.params.gamma * V_old[x, y])[..., np.newaxis]).reshape(self.position_action_shape)
            policy, V_new = greedy_policy(Q_value)
            V_new[self.target_position] = 0
            delta_V = V_new - V_old
            error = np.linalg.norm(delta_V)
            V_old = V_new

            if self.params.enable_sampling:
                self.params.sampler.log(k, {
                    'policy': policy.reshape(self.state_action_shape),
                })

        self.policy = policy.reshape(self.state_action_shape)
        self.S_value = V_old
