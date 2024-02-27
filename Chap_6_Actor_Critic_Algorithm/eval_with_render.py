"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 6:  RL example for lane keeping problem in a circle road
             Evaluate the trained agent with render
"""
# =================== load package ====================
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import lib
import gym
import torch
import torch.nn.functional as F

import lib  # noqa:
from lib import DeterminePolicy, StochasticPolicy, StateValue, ActionValue
from lib import policy_evaluate
from lib import self_plot
from pytorch_lightning import seed_everything

# ================== Set random seed ====================
seed_everything(0)

# ===================== get args ========================
parse = argparse.ArgumentParser()
parse.add_argument("--det_act", action="store_true", default=True)
parse.add_argument("--sto_act", action="store_true")
parse.add_argument("--sto_sta", action="store_true")
parse.add_argument("--path", type=str, default='./Results_dir/det_act/actor.pth')
args = parse.parse_args()

# ==================== Set log path ====================
if args.det_act is True:
    path = './Results_dir/det_act/actor.pth'
elif args.sto_act is True:
    path = './Results_dir/sto_act/actor.pth'
elif args.sto_sta is not True:
    path = './Results_dir/sto_sta/actor.pth'
else:
    raise ValueError

if args.path is not None:
    path = args.path

# ============ Initial environment and agent =============
env = gym.make('Lanekeeping-v0')
S_DIM = env.observation_space.shape[0]  # dimension of state space
A_DIM = env.action_space.shape[0]  # dimension of action space
if args.det_act is True:
    actor = DeterminePolicy(S_DIM, A_DIM)
else:
    actor = StochasticPolicy(S_DIM, A_DIM)

# load state_dict
action_state_dict = torch.load(path)

actor.load_state_dict(action_state_dict)

s = env.reset()
total_return = 0

for step in range(400):
    _, a = actor.choose_action(s)  # choose a action
    s, r, done, _ = env.step(a[0])
    total_return += r
    print("step = {:d} | r = {:5.3f} | R = {:5.3f} | done = {:d} | ".format(step, r, total_return, done), end="")
    print("s = [{:4.2f},{:4.2f},{:4.2f},{:4.2f},{:4.2f}]".format(s[0],s[1],s[2],s[3],s[4]), end="")
    print("a = [{:4.2f},{:4.2f}]".format(a[0][0],a[0][1]))
    env.render()

