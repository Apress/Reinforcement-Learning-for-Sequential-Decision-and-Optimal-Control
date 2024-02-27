"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

from math import inf
from typing import Tuple
import numba

import numpy as np
from tqdm import trange

from . import BaseGridAgent
from .. import epsilon_greedy_policy
from .njit import position_to_state, state_to_position, move_rand, move

class MCGridAgent(BaseGridAgent):
    def __init__(self, *args, **kwargs):
        super(MCGridAgent, self).__init__(*args, **kwargs)
        if self.params.max_step_per_episode is None:
            self.params.max_step_per_episode = 4 * (self.grid_size[0] + self.grid_size[1])
        self.Q_value = np.zeros(self.state_action_shape)
        self.S_value = np.zeros(self.grid_area)

    def train(self):
        """
        Monte Carlo PEV-PIM loop.
        """
        error = inf

        if self.params.enable_sampling:
            self.params.sampler.log(0, {
                'policy': self.policy,
                'S_value': self.S_value
            })

        iterator = trange(1, self.params.MC_max_iteration + 1, desc=self.params.name)
        for k in iterator:
            if self.params.MC_with_Q_initialization and error <= self.params.MC_Q_error_threshold:
                iterator.close()
                print("Early stop due to Q error threshold")
                break
            first_visit_stat = np.zeros(self.state_action_shape)
            every_visit_stat = np.zeros(self.grid_area)
            average_g_t = np.zeros(self.state_action_shape)

            # PEV
            for episode in range(self.params.max_episodes):
                g_t, s_index, route = self.episode()
                first_visit_stat += s_index > 0
                average_g_t += g_t

                current_episode = self.params.max_episodes * (k - 1) + episode + 1
                self.params.sampler.log(current_episode, {
                    'route_length': len(route)
                }, irregular=True)

                if self.params.enable_sampling and current_episode < 20:
                    # hacky code to avoid exploding log size...
                    # seeking better way
                    self.params.sampler.log(current_episode, {
                        'route': route
                    }, irregular=True)

                if self.params.enable_sampling and k == 1 and episode < 100:
                    # hacky code to avoid exploding log size...
                    # seeking better way
                    every_visit_stat[route] += 1
                    every_visit_stat[self.target_state] = 0
                    self.params.sampler.log(current_episode, {
                        'action_statistics': every_visit_stat
                    })

            first_visit_mask: np.ndarray = first_visit_stat > 0
            average_g_t[first_visit_mask] /= first_visit_stat[first_visit_mask]

            if self.params.MC_with_Q_initialization:
                new_Q_value = self.Q_value.copy()
                new_Q_value[first_visit_mask] = new_Q_value[first_visit_mask] * (1 - self.params.lamda) + average_g_t[first_visit_mask] * self.params.lamda
                error = np.max(np.abs(new_Q_value - self.Q_value))
                self.Q_value = new_Q_value
            else:
                self.Q_value = average_g_t

            # PIM
            self.policy, self.S_value = epsilon_greedy_policy(self.Q_value, self.params.epsilon)

            if self.params.enable_sampling:
                self.params.sampler.log(self.params.max_episodes*k, {
                    'policy': self.policy,
                    'S_value': self.S_value
                })

    def episode(self, start_position: Tuple[int, int] = None):
        """
        Simulate an episode. Calculate G_t.
        :return: Tuple of (G_t, S_index, Route)
        """
        start_state = self.generate_start_state() if start_position is None else self.position_to_state(start_position)
        return self._episode(start_state, self.state_action_shape, self.params.max_step_per_episode, self.params.gamma, self.actions, self.action_delta, self.policy, self.agent_dynamics, self.grid_size, self.params.reward, self.target_state)

    @staticmethod
    @numba.njit
    def _episode(start_state, state_action_shape, max_step_per_episode, gamma, actions, action_delta, policy, agent_dynamics, grid_size, rewards, target_state):
        while True:  # Until a successful route within limited steps
            s_index = np.zeros(state_action_shape)
            G_t = np.zeros(state_action_shape)

            current_state = start_state
            route = [current_state]
            for step in range(max_step_per_episode):
                next_state, action, reward, done = move_rand(current_state, actions, action_delta, policy, agent_dynamics, grid_size, rewards, target_state)
                route.append(next_state)

                if s_index[current_state, action] == 0:
                    s_index[current_state, action] = step + 1

                G_t += (s_index > 0) * np.power(gamma, step + 1 - s_index) * reward  # Refer to (3-2)

                if done:
                    return G_t, s_index, route

                current_state = next_state
