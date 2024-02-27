"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL parameters

"""

class RLParams:
    """
    Class for parameters used in reinforcement learning
    """

    def __init__(self, name:str, 
                 alpha:float=0.1, epsilon:float=0.95, gamma:float=0.9, lamda:float=0.1, maxIteration:int=1000,
                 maxEpisode:int=10000, PEVsteps:int=1, logPeriod:int=2000):
        """
        Generic parameters

        :param maxEpisode: Maximum number of episodes.
                             In MC, it's per iteration (policy evaluation) or use `episodes_per_iter`.
                             In TD, it's used in the top level loop.
        :param alpha: Learning rate, used in some algorithms.
        :param maxIteration: Maximum number of PEV-PIM Iteration, used in MC and DP.

        Parameters for long-term return

        :param gamma: Discount factor of the long-term return (2-9)

        Parameters for epsilon-greedy policy

        :param epsilon: Small probability of epsilon-greedy policy (2-6)

        Parameters for Q Learning & Sarsa algorithm

        :param logPeriod: Record the trainning process every [logPeriod] episodes

        Parameters for Sarsa algorithm

        :param PEVsteps: Number of {S,A} pair in PEV

        """
        assert 0 <= epsilon <= 1, "Epsilon shall be between 0 and 1"
        assert 0 <= gamma <= 1, "Gamma shall be between 0 and 1"
        assert logPeriod > 0, "LogPeriod shall be positive"
        assert maxIteration > 0, "maxIteration shall be positive"
        assert maxEpisode > 0, "maxEpisode shall be positive"
        assert PEVsteps > 0, "PEVsteps shall be positive"

        self.name = name
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lamda = lamda
        self.maxIteration = maxIteration
        self.maxEpisode=maxEpisode
        self.PEVsteps = PEVsteps
        self.logPeriod = logPeriod
