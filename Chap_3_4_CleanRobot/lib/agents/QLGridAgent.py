"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

import numpy as np
import math

from tqdm import trange

from . import BaseGridAgent
from .. import greedy_policy


class QLGridAgent(BaseGridAgent):
    def __init__(self, *args, **kwargs):
        super(QLGridAgent, self).__init__(*args, **kwargs)
        if self.params.max_step_per_episode is None:
            self.params.max_step_per_episode = 500000
        # self.disp_episode = math.ceil(self.params.max_episodes/100)
        self.Q_value = np.zeros(self.state_action_shape)
        self.S_value = np.zeros(self.grid_area)

    def train(self):
        """
        Q-learning PEV-PIM loop.
        """
        if self.params.enable_sampling:
            self.params.sampler.log(0, {
                'policy': self.policy,
                'S_value': self.S_value
            })

        for k in trange(self.params.max_episodes, desc=self.params.name):
            current_state = self.generate_start_state()

            for step in range(self.params.max_step_per_episode):
                next_state, action, reward, done = self.move(current_state)
                self.Q_value[current_state][action] += self.params.alpha * (reward + self.params.gamma * np.max(self.Q_value[next_state]) - self.Q_value[current_state][action])
                if done:
                    break
                current_state = next_state

            if self.params.enable_sampling:
                # intermediate policy is not used
                new_policy, new_S_value = greedy_policy(self.Q_value)
                self.params.sampler.log(k, {
                    'policy': new_policy,
                    'S_value': new_S_value
                })
                self.params.sampler.log(k, {
                    'route_length': step + 1
                }, irregular=True)

            self.Q_value *= self.params.QL_init_coefficient

        self.policy, self.S_value = greedy_policy(self.Q_value)
