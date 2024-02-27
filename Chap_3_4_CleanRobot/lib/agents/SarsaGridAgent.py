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
from .. import epsilon_greedy_policy, RingBuffer


class SarsaGridAgent(BaseGridAgent):
    def __init__(self, *args, **kwargs):
        super(SarsaGridAgent, self).__init__(*args, **kwargs)
        if self.params.max_step_per_episode is None:
            self.params.max_step_per_episode = 500000  # avoid hard-code?
        # self.disp_episode = math.ceil(self.params.max_episodes / 100)
        self.state_buffer = RingBuffer(self.params.Sarsa_TD_steps)
        self.action_buffer = RingBuffer(self.params.Sarsa_TD_steps)
        self.reward_buffer = RingBuffer(self.params.Sarsa_TD_steps, np.float32)
        self.Q_value = np.zeros(self.state_action_shape)
        self.S_value = np.zeros(self.grid_area)

    def train(self):
        """
        Sarsa train loop.
        """
        PEV_steps = 0

        if self.params.enable_sampling:
            self.params.sampler.log(0, {
                'policy': self.policy,
                'S_value': self.S_value
            })

        for k in trange(self.params.max_episodes, desc=self.params.name):
            current_state = self.generate_start_state()
            action = np.random.choice(self.actions, p=self.policy[current_state])

            for t in range(self.params.max_step_per_episode, 0, -1):
                next_state, _, reward, done = self.move(current_state, action)
                next_action = np.random.choice(self.actions, p=self.policy[next_state]) if not done and t > 1 else None

                self.log_sar(current_state, action, reward)

                if self.state_buffer.filled():
                    idx = np.arange(self.params.Sarsa_TD_steps)
                    G = np.sum(np.power(self.params.gamma, idx) * self.reward_buffer.get())
                    if next_action is not None:
                        G += np.power(self.params.gamma, self.params.Sarsa_TD_steps) * self.Q_value[next_state][next_action]

                    Q_position, Q_action = self.state_buffer.head(), self.action_buffer.head()
                    self.Q_value[Q_position][Q_action] += self.params.alpha * (G - self.Q_value[Q_position][Q_action])
                    PEV_steps += 1

                    if PEV_steps == self.params.Sarsa_PEV_steps:
                        self.policy, self.S_value = epsilon_greedy_policy(self.Q_value, self.params.epsilon)
                        PEV_steps = 0

                if next_action is None:
                    self.reset_sar()
                    break

                action = next_action
                current_state = next_state

            if self.params.enable_sampling:
                self.params.sampler.log(k, {
                    'route_length': self.params.max_step_per_episode - t + 1
                }, irregular=True)

                self.params.sampler.log(k, {
                    'policy': self.policy,
                    'S_value': self.S_value
                })

    def log_sar(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def reset_sar(self):
        self.state_buffer.reset()
        self.action_buffer.reset()
        self.reward_buffer.reset()
