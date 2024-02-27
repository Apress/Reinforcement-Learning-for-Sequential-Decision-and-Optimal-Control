"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Yuxuan Jiang

Description: Chapter 3 & 4: Plot figures for DP, MC, Sarsa, and Q-learning algorithms

"""


# =================== load package ====================
import math

import matplotlib.pyplot as plt
import numpy as np

from lib import RLParams, repeat
from lib.agents import BaseGridAgent, DPGridAgent, MCGridAgent, QLGridAgent, SarsaGridAgent
from lib.plot import plot_policy, plot_value, plot_route, plot_statistics, plot_rms, plot_reward_pre, \
    plot_route_animated, config, cm2inch
from lib.plot.config import default_cfg
from matplotlib import rc


# ================ General Configs for plot ================
plt.rcParams['font.size'] = '6'
common_problem_defs = {
    'grid_size': 6, 'target_position': (0, 5)
}
OUTPUT_ROOT_DIR = 'figures'
OUTPUT_FORMAT = 'png'
PLT_CONF = default_cfg
DIR_ROOT_NAME = 'plot_data'
AGENT_NAME = 'agent'
REWARD_NAME = 'reward'


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
