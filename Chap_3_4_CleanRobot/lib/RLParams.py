"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4:  parameters for reinforcement learning

"""

from math import inf

from .Sampler import Sampler


class RLParams:
    """
    Class for parameters used in reinforcement learning
    """

    def __init__(self, name: str,
                 reward = (-1, 9), alpha: float = 0.1, epsilon: float = 0.05, gamma: float = 0.9, lamda: float = 0.9,
                 max_episodes: int = 1, episodes_per_iter: int = None, max_step_per_episode: int = inf,
                 enable_sampling: bool = False, sampler: Sampler = None, sampler_options: dict = None,
                 MC_with_Q_initialization: bool = False, MC_max_iteration: int = 1, MC_Q_error_threshold: float = None,
                 Sarsa_TD_steps: int = 1, Sarsa_PEV_steps: int = 1,
                 QL_init_coefficient: float = 1,
                 DP_max_iteration: int = 1, DP_error_threshold: float = 0.01):
        """
        Generic parameters

        :param max_episodes: Maximum number of episodes.
                             In MC, it's per iteration (policy evaluation) or use `episodes_per_iter`.
                             In TD, it's used in the top level loop.
        :param reward: Learning reward, type may vary according to specific problems
        :param max_step_per_episode: Maximum number of steps in an episode
        :param alpha: Learning rate, used in some algorithms.
        :param lamda: coefficient for averaging policy

        Parameters for long-term return

        :param gamma: Discount factor of the long-term return (2-9)

        Parameters for epsilon-greedy policy

        :param epsilon: Small probability of epsilon-greedy policy (2-6)

        Parameters for data sampling

        :param enable_sampling: Enable sampling while training. Default to false.
        :param sampler: Sampler when sampling enabled.
        :param sampler_options: Options to pass to sampler constructor if no sampler provided

        Parameters for Monte Carlo algorithm

        :param MC_max_iteration: Maximum number of iterations
        :param MC_with_Q_initialization: Use Q-initialization or not (3.4.3)
        :param MC_Q_error_threshold: Error threshold for action value function between iterations, uses L-infinity,
                                     available when `MC_with_Q_initialization` is True

        Parameters for Sarsa algorithm

        :param Sarsa_TD_steps: Maximum number of iterations
        :param Sarsa_PEV_steps: Update policy every `Sarsa_PEV_steps` iterations

        Parameters for Q-Learning algorithm

        :param QL_init_coefficient: Coefficient to initialize Q-value

        Parameters for dynamic programming

        :param DP_max_iteration: dynamic programming max iteration
        :param DP_error_threshold: dynamic programming error threshold
        """
        assert 0 <= gamma <= 1, "Gamma shall be between 0 and 1"
        assert 0 <= epsilon <= 1, "Epsilon shall be between 0 and 1"
        assert max_episodes > 0, "Episodes per policy evaluation shall be positive"
        assert episodes_per_iter is None or episodes_per_iter > 0, "Maximum number of episodes in an iteration shall be positive"
        assert max_step_per_episode is None or max_step_per_episode > 0, "Maximum number of steps in an episode shall be positive"
        assert MC_max_iteration > 0, "Max iteration number shall be positive"
        assert MC_Q_error_threshold is None or MC_Q_error_threshold >= 0, "Error threshold for action value function shall be non-negative"
        assert Sarsa_TD_steps > 0, "TD step shall be positive"
        assert Sarsa_PEV_steps > 0, "PEV step shall be positive"
        assert QL_init_coefficient > 0, "Init coefficient shall be positive"
        assert DP_max_iteration > 0, "Max iteration number shall be positive"
        assert DP_error_threshold > 0, "Error threshold shall be positive"

        if enable_sampling and sampler is None:
            sampler = Sampler() if sampler_options is None else Sampler(**sampler_options)

        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.MC_max_iteration = MC_max_iteration
        if episodes_per_iter is not None:
            self.max_episodes = episodes_per_iter
        else:
            self.max_episodes = max_episodes
        self.reward = reward
        self.max_step_per_episode = max_step_per_episode
        self.enable_sampling = enable_sampling
        self.sampler = sampler
        self.MC_with_Q_initialization = MC_with_Q_initialization
        self.MC_Q_error_threshold = MC_Q_error_threshold
        self.Sarsa_TD_steps = Sarsa_TD_steps
        self.Sarsa_PEV_steps = Sarsa_PEV_steps
        self.QL_init_coefficient = QL_init_coefficient
        self.DP_max_iteration = DP_max_iteration
        self.DP_error_threshold = DP_error_threshold
