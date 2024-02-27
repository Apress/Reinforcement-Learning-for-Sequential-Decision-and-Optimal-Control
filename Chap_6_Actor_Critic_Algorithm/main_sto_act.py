"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 6:  RL example for lane keeping problem in a circle road
             Actor-Critic algorithm with stochastic policy and action value function

"""
# =================== load package ====================
import argparse
import os
from datetime import datetime

import numpy as np
import lib
import gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything

from lib.agent import StochasticPolicy, ActionValue
from lib.utils import policy_evaluate
from lib.idplot import self_plot

# ============= Setting hyper-parameters ===============
parse = argparse.ArgumentParser(description="Training parameters")
parse.add_argument("--lr_a", type=float, default=6e-4, help="learnint rate of actor")
parse.add_argument("--lr_c", type=float, default=6e-4, help="learnint rate of critic")
parse.add_argument("--gamma", type=float, default=0.97, help="parameter gamma")
parse.add_argument("--iter_num", type=int, default=4000, help="total number of training iteration")
parse.add_argument("--batch_size", type=int, default=256, help="minimum samples in a update batch")
parse.add_argument("--step_limit", type=int, default=128, help="maximum step the agent can run in the environment")
parse.add_argument("--seed", type=int, default=0, help="random seed")
parse.add_argument("--smoothing", action="store_true")
arg = parse.parse_args()

# ================== Set random seed ====================
seed_everything(arg.seed)

# ==================== Set log path ====================
log_dir = "./Results_dir/" + "sto_act/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(log_dir)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(log_dir, "tb"))

# ============ store important parameters ==============
with open(os.path.join(log_dir, "config.txt"), 'at') as f:
    f.write(str(arg))

# ============ Initial environment and agent =============
env = gym.make('Lanekeeping-v0')
S_DIM = env.observation_space.shape[0]  # dimension of state space
A_DIM = env.action_space.shape[0]  # dimension of action space

actor = StochasticPolicy(S_DIM, A_DIM)
critic = ActionValue(S_DIM + A_DIM, 1)

opt_actor = torch.optim.Adam(actor.parameters(), arg.lr_a)
opt_critic = torch.optim.Adam(critic.parameters(), arg.lr_c)

# ====================== Training ========================
r_plot = []
tars = []
for iter_index in range(arg.iter_num):
    time_step_batch = 0
    paths = []

    # roll out (s, a, s', r, done)
    while True:
        s = env.reset()
        states = []
        actions = []
        rewards = []
        time_step_ep = 0
        states_next = []
        terminals = []
        while True:
            states.append(s)
            a, _ = actor.choose_action(s)  # choose a action
            actions.append(a)
            s, r, done, _ = env.step(a[0])  # the environment outputs s', r and done.
            states_next.append(s)
            rewards.append(r)
            terminals.append(done)
            time_step_ep += 1
            if done or time_step_ep >= arg.step_limit:
                break

        path = {"states": np.array(states),
                "states_next": np.array(states_next),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "terminal": np.array(terminals)}

        paths.append(path)
        time_step_batch += len(path["reward"])
        if time_step_batch > arg.batch_size:
            break

    # compute date which is used in updating
    state_batch = np.concatenate([path["states"] for path in paths])
    action_batch = np.concatenate([path["action"] for path in paths])
    state_next_batch = np.concatenate([path["states_next"] for path in paths])
    terminals_batch = np.concatenate([path["terminal"] for path in paths])
    rewards_batch = np.concatenate([path["reward"] for path in paths])

    # Actor loss & update
    state_tensor = torch.Tensor(state_batch)
    mu, sig, dist = actor(state_tensor)
    pi = mu + torch.randn_like(mu, dtype=torch.float32) * sig
    q_tensor = critic(state_tensor, pi)
    a_loss = torch.mean(-q_tensor)
    opt_actor.zero_grad()
    a_loss.backward()
    opt_actor.step()

    # Critic loss & update
    for _ in range(20):
        a_tensor = torch.Tensor(action_batch).reshape(-1, A_DIM)
        q_tensor = critic(state_tensor, a_tensor)
        state_next_tensor = torch.Tensor(state_next_batch)
        reward_tensor = torch.Tensor(rewards_batch).reshape(-1, 1)
        terminal_tensor = torch.Tensor(terminals_batch).reshape(-1, 1)
        pi_next, _, _ = actor(state_next_tensor)
        pi_next = pi_next.reshape(-1, A_DIM)
        q_next = critic(state_next_tensor, pi_next)
        q_target = reward_tensor + arg.gamma * (1 - terminal_tensor) * q_next
        q_loss = F.mse_loss(q_tensor, q_target.detach())
        opt_critic.zero_grad()
        q_loss.backward()
        opt_critic.step()

    total_reward = np.sum([path["reward"].sum() for path in paths])
    average_reward = total_reward / time_step_batch

    writer.add_scalar("tag/reward", average_reward, iter_index)
    writer.add_scalar("tag/a_loss", a_loss.item(), iter_index)
    writer.add_scalar("tag/c_loss", q_loss.item(), iter_index)

    # average reward smoothing
    if arg.smoothing == True:
        if len(r_plot) == 0:
            r_plot.append(average_reward)
        else:
            r_plot.append(average_reward * 0.5 + r_plot[-1] * 0.5)
    else:
        r_plot.append(average_reward)
    
    # evaluate policy and record the results
    if iter_index % 5 == 0:
        tar,start = policy_evaluate(env, actor, "sto")
        tars.append(tar)
        writer.add_scalar("tag/return", tar, iter_index)
        log_trace = "Iteration: {:3d} |" \
                    "Average reward: {:7.3f} |" \
                    "Average return: {:7.3f}".format(iter_index, average_reward, tar)
        print(log_trace)
    
    if iter_index % 5 == 0:
        actor.save_parameter(log_dir, iter_index)

# ================= save result =======================
np.save(os.path.join(log_dir, "r.npy"), r_plot)
actor.save_parameter(log_dir)
np.save(os.path.join(log_dir, "tar.npy"), tars)
