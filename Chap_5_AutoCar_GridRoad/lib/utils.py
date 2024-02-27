"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 5:  Functions for Example AutoCar
"""


import numpy as np
import os

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


default_cfg = AttrDict()

default_cfg.fig_size = (8.5, 5.5)
default_cfg.fig_size_squre = (6.5, 6.5)
default_cfg.dpi = 600
default_cfg.pad = 0.2
default_cfg.figsize_scalar = 1
default_cfg.tick_size = 8
default_cfg.linewidth = 2
default_cfg.tick_label_font = 'Times New Roman'
default_cfg.legend_font = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
default_cfg.label_font = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}


def makedir(ABS_DIR, foldername):
    try:
        os.mkdir('/'.join([ABS_DIR, foldername]))
    except FileExistsError:
        pass
    return


def headlocation(s):
    s_h = np.array([0, 0, 0])
    if s[2] == 0:
        s_h = s + np.array([0, 1, 0])
    elif s[2] == 1:
        s_h = s + np.array([1, 1, 0])
    elif s[2] == 2:
        s_h = s + np.array([1, 0, 0])
    elif s[2] == 3:
        s_h = s + np.array([1, -1, 0])
    elif s[2] == 4:
        s_h = s + np.array([0, -1, 0])

    return s_h


def isoutside(s, boundary):
    # return 0 if the car is outside of the lane else 1

    s_h = headlocation(s)
    if s_h[0] < len(boundary[0, :]) and s_h[0] >= 0:
        myBoole = (s[1] <= boundary[0, s[0]]) * (s[1] >= boundary[1, s[0]]) * \
                  (s_h[1] <= boundary[0, s_h[0]]) * (s_h[1] >= boundary[1, s_h[0]]) * \
                  (s[2] >= 0) * (s[2] <= 4)
    elif s_h[0] >= len(boundary[0, :]):  # Considering the condition that the tail is at the terminal
        myBoole = (s[1] <= boundary[0, s[0]]) * (s[1] >= boundary[1, s[0]]) * \
                  (s[2] >= 0) * (s[2] <= 4)
    else:
        myBoole = 0
    return myBoole


def nextstate(s, action):
    # s: state, s[0]-position x; s[1]-position y; s[2]-heading direction h
    # action: left, right, keep
    s_next = s.copy()

    ## change the heading
    if action == 0:  # left
        s_next[2] = s[2] - 1
    elif action == 1:  # right
        s_next[2] = s[2] + 1
    elif action == 2:  # keep
        s_next[2] = s[2]

    if s_next[2] == 0:
        s_next[0] = s[0]
        s_next[1] = s[1] + 1
    elif s_next[2] == 1:
        s_next[0] = s[0] + 1
        s_next[1] = s[1] + 1
    elif s_next[2] == 2:
        s_next[0] = s[0] + 1
        s_next[1] = s[1]
    elif s_next[2] == 3:
        s_next[0] = s[0] + 1
        s_next[1] = s[1] - 1
    elif s_next[2] == 4:
        s_next[0] = s[0]
        s_next[1] = s[1] - 1
    return s_next


def precondition(s_size, a_size, laneBoundary):
    # s_size: state size, list object with 3 elements
    # a_size: action size, int
    size_s_a = [s_size[0], s_size[1], s_size[2], a_size]
    deadaction_flag = np.zeros(size_s_a, dtype=int)  # signal of the action that causing the car out of the lane
    deadstate_flag = np.zeros(s_size, dtype=int)  # Signal of the state that cannot find any feasbile action
    for i in range(5):
        for s1 in range(s_size[0] - 1):
            for s2 in range(s_size[1]):
                for s3 in range(s_size[2]):
                    for a in range(a_size):
                        s_now = np.array([s1, s2, s3])
                        error_now = isoutside(s_now, laneBoundary)
                        if error_now == 0:
                            deadstate_flag[s1, s2, s3] = 1
                            break
                        s_next = nextstate(s_now, a)
                        # print(s_next)
                        error_next = isoutside(s_next, laneBoundary)
                        # print(error_next)
                        # print(deadstate_flag[s_next[0],s_next[1],s_next[2]])
                        # print(deadstate_flag.shape)
                        if (error_next == 0) or (deadstate_flag[s_next[0], s_next[1], s_next[2]] == 1):
                            deadaction_flag[s1, s2, s3, a] = 1
                    if np.min(deadaction_flag[s1, s2, s3, :]) == 1:
                        deadstate_flag[s1, s2, s3] = 1

    return deadstate_flag, deadaction_flag


def precondition2(s_size, a_size, laneBoundary):
    size_s_a = s_size.copy()
    size_s_a.append(a_size)
    deadaction_flag = np.zeros(size_s_a)
    deadstate_flag = np.zeros(s_size)

    for idx in np.ndindex(deadstate_flag[:-1, ...].shape):
        for a in range(a_size):
            s_now = np.array(idx)
            error_now = isoutside(s_now, laneBoundary)
            if error_now == 0:
                deadstate_flag[idx] = 1
                break
            s_next = nextstate(s_now, a)
            error_next = isoutside(s_next, laneBoundary)
            if error_next == 0 or deadstate_flag[idx] == 1:
                deadaction_flag[idx][a] = 1
        if np.min(deadaction_flag[idx]) == 1:
            deadstate_flag[idx] = 1

    return deadstate_flag, deadaction_flag


def reward(s, action, s_next):
    # Reward function
    # 原matlab函数中 形参 s 位置为 ~（缺省）

    # action = keep, reward = 0 else reward = -1
    if action == 0 or action == 1:
        R_steer = -1
    else:
        R_steer = 0

    if s_next[2] == 0 or s_next[2] == 2 or s_next[2] == 4:
        R_move = -1
    else:
        R_move = -1.4

    return R_move + R_steer


def get_action(Q, s, epsilon, deadaction_flag):
    action_aval = []
    for a in range(3):
        if deadaction_flag[s[0], s[1], s[2], a] == 0:
            action_aval.append(a)
    choice = np.random.rand()
    if choice < 1 - epsilon:
        Ql = list(Q[s[0], s[1], s[2], action_aval])
        # print(action_aval)
        num = Ql.index(max(Ql))
        action = action_aval[num]
    else:
        num = np.random.randint(len(action_aval))
        action = action_aval[num]
    return action


def preconditionMC(s_size, a_size, laneBoundary):
    deadstate_flag, deadaction_flag = precondition(s_size, a_size, laneBoundary)
    unfeasible_start = np.zeros_like(deadstate_flag[:-1, ...])
    next_state_m1 = np.zeros_like(deadaction_flag[:-1, ...])
    next_state_m2 = np.zeros_like(deadaction_flag[:-1, ...])
    next_state_m3 = np.zeros_like(deadaction_flag[:-1, ...])
    s_h_m1 = np.zeros_like(deadstate_flag)
    s_h_m2 = np.zeros_like(deadstate_flag)
    s_h_m3 = np.zeros_like(deadstate_flag)

    s_h = np.zeros(3, dtype=int)
    for idx in np.ndindex(deadstate_flag[:-1, ...].shape):
        s = np.array(idx)
        s_h = headlocation(s)
        s_h_m1[idx] = s_h[0]
        s_h_m2[idx] = s_h[1]
        s_h_m3[idx] = s_h[2]
        if deadstate_flag[idx] == 1 or s_h[0] == s_size[0] - 1:
            unfeasible_start[idx] = 1

    for idx in np.ndindex(deadstate_flag.shape):
        s = np.array(idx)
        s_h = headlocation(s)
        s_h_m1[idx] = s_h[0]
        s_h_m2[idx] = s_h[1]
        s_h_m3[idx] = s_h[2]

    s = np.zeros(3, dtype=int)
    reward_m = np.zeros(deadaction_flag[:-1, ...].shape)
    for idx in np.ndindex(deadaction_flag[:-1, ...].shape):
        s = np.array(idx[:-1])
        action = idx[3]
        s_next = nextstate(s, action)
        next_state_m1[idx] = s_next[0]
        next_state_m2[idx] = s_next[1]
        next_state_m3[idx] = s_next[2]
        reward_m[idx] = reward(1, action, s_next)

    return deadstate_flag, deadaction_flag, unfeasible_start, next_state_m1, next_state_m2, next_state_m3, s_h_m1, s_h_m2, s_h_m3, reward_m


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)
