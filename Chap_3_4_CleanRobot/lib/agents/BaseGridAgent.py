"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

from math import inf
from typing import Union, Tuple
import itertools

import numpy as np

from .. import RLParams, is_valid_index
from .njit import position_to_state, state_to_position, move_rand, move

class BaseGridAgent:
    default_actions = (0, 1, 2, 3)
    default_agent_dynamics = np.array((
        (0.8, 0.1, 0.0, 0.1),
        (0.1, 0.8, 0.1, 0.0),
        (0.0, 0.1, 0.8, 0.1),
        (0.1, 0.0, 0.1, 0.8)
    ))
    default_action_delta = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1)
    )

    def __init__(self, grid_size: Union[int, Tuple[int, int]], rl_params: RLParams, target_position: Tuple[int, int] = None,
                 actions: Tuple[int, ...] = None, agent_dynamics: Tuple[Tuple[float, ...], ...] = None,
                 action_delta: Tuple[Tuple[int, int], ...] = None, policy=None):
        """
        Base grid agent
        :param grid_size: Pass int as a short-hand for square; pass (row, col) tuple for rectangle
        :param rl_params: Reinforcement learning parameters
        :param target_position: Target position, defaults to (row-1, col-1)
        :param actions: Actions, default to four directions
        :param agent_dynamics: Matrix with shape (len(actions), len(actions))
        :param action_delta: Matrix with shape (len(actions), 2)
        :param policy: Grid agent policy
        """
        # Normalize & validate parameters
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        assert grid_size[0] > 0 and grid_size[1] > 0, "Grid size shall be positive"
        if target_position is None:
            target_position = (0, grid_size[1] - 1)  # right bottom
        assert is_valid_index(target_position, grid_size), "Target position outside grid"
        if actions is None:
            actions = BaseGridAgent.default_actions
        if agent_dynamics is None:
            agent_dynamics = BaseGridAgent.default_agent_dynamics
        if action_delta is None:
            action_delta = BaseGridAgent.default_action_delta
        assert len(rl_params.reward) == 2

        # Assign class fields
        self.grid_size = grid_size
        self.grid_area = grid_size[0] * grid_size[1]
        self.params = rl_params
        self.target_position = target_position
        self.target_state = self.position_to_state(target_position)
        self.actions = actions
        self.agent_dynamics = agent_dynamics
        self.action_delta = action_delta
        self.state_action_shape = (self.grid_area, len(self.actions))
        self.position_action_shape = self.grid_size + (len(self.actions),)
        if policy is None:
            policy = 0.25 * np.ones(self.state_action_shape)
        self.policy = policy

    def run(self, start_position: Tuple[int, int] = None, start_state: int = None, as_position: bool = True):
        """
        Simulate an episode.

        :param start_state: Start state
        :param start_position: Start position, default to a randomly generated position
        :param as_position: Convert route to list of position, default to True
        :return: Tuple of (route, total_reward, successful)
        """
        convert_route = (lambda r: list(zip(*self.state_to_position(r)))) if as_position else (lambda r: r)
        if start_state is not None:
            current_state = start_state
        else:
            current_state = self.generate_start_state() if start_position is None else self.position_to_state(start_position)
        total_reward = 0
        route = [current_state]
        for step in itertools.count():
            next_state, _, reward, done = self.move(current_state)
            total_reward += reward
            route.append(next_state)
            # print("RL Iteration = %d" % step)
            if done:
                return convert_route(route), total_reward, True
            if step >= self.params.max_step_per_episode:
                return convert_route(route), total_reward, False
            current_state = next_state

    def traverse(self, repeat: int = 1):
        """
        Set start point at each cell of the grid. Pretty slow.
        """
        rewards = [[self.run(start_state=i, as_position=False)[1] for _ in range(repeat)] for i in range(self.grid_area) if i != self.target_state]
        return np.mean(rewards)

    def traverse_optimized(self, repeat: int = 1, mode: str = 'greedy'):  # 'greedy', 'epsilon', 'equal'
        """
        Still the simulation way. Like `traverse`, but used a random pool to boost 10 times performance.
        TODO: The random pool approach could be adapted to other simulation agents to improve performance,
        TODO: providing it uses greedy or epsilon-greedy policy.
        """
        if mode == 'equal':
            raise NotImplementedError
        else:
            policy = self.get_dominant_policy()
            real_pool = np.random.choice([0, 1, 2, 3], size=(repeat * 50 * self.grid_area,), p=(0.8, 0.1, 0, 0.1))
            real_map = np.array(((0, 1, 2, 3),
                                (1, 2, 3, 0),
                                (2, 3, 0, 1),
                                (3, 0, 1, 2)))
            count = 0

            def get_real_action(action, pos):
                nonlocal count
                delta = BaseGridAgent.default_action_delta[real_map[action, real_pool[count]]]
                count += 1
                next_pos = (pos[0] + delta[0], pos[1] + delta[1])

                if not is_valid_index(next_pos, self.grid_size):
                    return pos, action, -1, False

                done = next_pos == self.target_position
                reward = self.params.reward[1 if done else 0]
                return next_pos, action, reward, done

            if mode == 'greedy':
                def run(state):
                    total_reward = 0
                    pos = self.state_to_position(state)
                    for step in itertools.count():
                        next_pos, _, reward, done = get_real_action(policy[pos], pos)
                        total_reward += reward
                        if done:
                            return total_reward
                        if step >= self.params.max_step_per_episode:
                            return total_reward
                        pos = next_pos
                rewards = [[run(i) for _ in range(repeat)] for i in range(self.grid_area) if i != self.target_state]
                return np.mean(rewards)
            else:
                eps_pool = np.random.choice([0, 1, 2, 3], size=(repeat * 50 * self.grid_area,), p=(1-self.params.epsilon*3/4, self.params.epsilon/4, self.params.epsilon/4, self.params.epsilon/4))
                count_eps = 0

                def get_action(action):
                    nonlocal count_eps
                    a = real_map[action, eps_pool[count_eps]]
                    count_eps += 1
                    return a

                def run(state):
                    total_reward = 0
                    pos = self.state_to_position(state)
                    for step in itertools.count():
                        next_pos, _, reward, done = get_real_action(get_action(policy[pos]), pos)
                        total_reward += reward
                        if done:
                            return total_reward
                        if step >= self.params.max_step_per_episode:
                            return total_reward
                        pos = next_pos
                rewards = [[run(i) for _ in range(repeat)] for i in range(self.grid_area) if i != self.target_state]
                return np.mean(rewards)

    def traverse_iterative(self, max_iterations: int = 50, tolerance: float = 0.0001):
        """
        Iterative way. Huge performance gain.
        TODO: Is divergence avoidable without `max_iterations`?

        :param tolerance: L2 norm
        :param max_iterations: had to add it because some bad/initial policies may lead to diverge
        """
        _x, _y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing='ij')
        x = np.tile(_x[..., np.newaxis], (1, 1, len(self.actions)))
        x += np.array([1, 0, -1, 0]).reshape((1, 1, -1))
        x[x < 0] = 0
        x[x >= self.grid_size[0]] = self.grid_size[0] - 1

        y = np.tile(_y[..., np.newaxis], (1, 1, len(self.actions)))
        y += np.array([0, 1, 0, -1]).reshape((1, 1, -1))
        y[y < 0] = 0
        y[y >= self.grid_size[1]] = self.grid_size[1] - 1

        step_reward = np.zeros(self.position_action_shape)
        step_reward[:] = self.params.reward[0]
        step_reward[np.logical_and(x == self.target_position[0], y == self.target_position[1])] = self.params.reward[1]

        rewards = np.zeros(self.grid_size)
        real_policy = self.policy.reshape(self.position_action_shape) @ self.agent_dynamics

        expect_step_reward = np.sum(np.multiply(real_policy, step_reward), axis=2)
        expect_step_reward[self.target_position] = 0

        error = inf
        k = 0
        while k < max_iterations and error > tolerance:
            k += 1
            new_rewards = expect_step_reward + np.sum(np.multiply(real_policy, rewards[x, y]), axis=2)
            new_rewards[self.target_position] = 0
            error = np.linalg.norm(new_rewards - rewards)
            rewards = new_rewards

        return np.sum(rewards) / (self.grid_area - 1)

    def position_to_state(self, position: Tuple[int, int]):
        """
        Convert position to state.

        :param position: Grid position
        :return: Grid state, row-major
        """
        return np.ravel_multi_index(position, self.grid_size)

    def state_to_position(self, state: int):
        """
        Convert position to state.

        :param state: Grid state, row-major
        :return: Grid position
        """
        return np.unravel_index(state, self.grid_size)

    def move(self, current_state: int, action: int = None):
        """
        Environment model (2-1)

        :param current_state: Current state
        :param action: Action chosen, default to `np.choice` by policy
        :return: Tuple of (next_state, action, reward, done)
        """
        if action is None:
            return move_rand(current_state, self.actions, self.action_delta, self.policy, self.agent_dynamics, self.grid_size, self.params.reward, self.target_state)
        else:
            return move(current_state, action, self.actions, self.action_delta, self.agent_dynamics, self.grid_size, self.params.reward, self.target_state)

    def generate_start_state(self):
        """
        Generate start state randomly.

        :return: Starting state
        """
        start_state = np.random.randint(0, self.grid_area - 1)

        if start_state == self.target_state:
            # Only happens when the target isn't grid_area, i.e. the target position isn't (rol-1, col-1)
            return self.grid_area - 1

        return start_state

    def get_dominant_policy(self):
        return np.argmax(self.policy, axis=-1).reshape(self.grid_size)
