"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 6:  RL example for lane keeping problem in a circle road
             Actor-Critic algorithm with stochastic policy and state value function

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
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything

from lib import StochasticPolicy, StateValue
from lib import policy_evaluate
from lib import self_plot

# ============= Setting hyper-parameters ===============
parse = argparse.ArgumentParser(description="Training parameters")
parse.add_argument("--norm", action="store_true", help="normalization of bootstrap", default=True)
parse.add_argument("--baseline", action="store_true", help="add baseline")
parse.add_argument("--lr_a", type=float, default=2e-3, help="learnint rate of actor")
parse.add_argument("--lr_c", type=float, default=2e-3, help="learnint rate of critic")
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
log_dir = "./Results_dir/" + ("sto_sta_baseline/" if arg.baseline else "sto_sta/") + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
critic = StateValue(S_DIM, 1)

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
        states_next = []
        actions = []
        rewards = []
        time_step_ep = 0
        terminals = []
        while True:
            states.append(s)
            a, _ = actor.choose_action(s)  # choose a action
            actions.append(a[0])
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
    v_predicts = []
    v_targets = []

    # computing episode by episode
    for path in paths:
        reward_ep = path["reward"].reshape(-1, 1)
        
        v_predicts_ep = critic(torch.Tensor(path["states"])).detach().numpy()
        
        v_predicts_next_ep = critic(torch.Tensor(path["states_next"])).detach().numpy()

        bootstrap_ep = reward_ep + arg.gamma * (1 - path["terminal"].reshape(-1, 1)) * v_predicts_next_ep

        v_predicts.append(v_predicts_ep)
        v_targets.append(bootstrap_ep)

    # combined all the episode together
    state_batch = np.concatenate([path["states"] for path in paths])
    action_batch = np.concatenate([path["action"] for path in paths])
    state_next_batch = np.concatenate([path["states_next"] for path in paths])
    terminals_batch = np.concatenate([path["terminal"] for path in paths])
    rewards_batch = np.concatenate([path["reward"] for path in paths])
    v_predict_batch = np.concatenate(v_predicts)
    v_target_batch = np.concatenate(v_targets)
    
    state_tensor = torch.Tensor(state_batch)
    action_tensor = torch.Tensor(action_batch)
    v_predict_tensor = torch.Tensor(v_predict_batch)
    state_next_tensor = torch.Tensor(state_next_batch)
    reward_tensor = torch.Tensor(rewards_batch).reshape(-1, 1)
    terminal_tensor = torch.Tensor(terminals_batch).reshape(-1, 1)
    
    # Critic loss & update
    for _ in range(20):
        v_next_tensor = critic(state_next_tensor)
        v_target_tensor = reward_tensor + arg.gamma * (1 - terminal_tensor) * v_next_tensor
        v_predict_tensor = critic(state_tensor.detach())
        v_loss = F.mse_loss(v_predict_tensor, v_target_tensor.detach())
        opt_critic.zero_grad()
        v_loss.backward() 
        opt_critic.step()
    
    # Actor loss & update
    if arg.baseline is True:
        bootstrap = v_target_tensor - critic(state_tensor)
    else:
        bootstrap = v_target_tensor
    
    if arg.norm is True:
        bootstrap = (bootstrap - bootstrap.mean()) / (bootstrap.std() + 1e-8) * 25
    
    _, _, dist = actor(state_tensor.detach())
    log_pi = dist.log_prob(action_tensor.detach()).unsqueeze(1)
    a_loss = torch.mean(-log_pi * bootstrap.detach())
    opt_actor.zero_grad()
    a_loss.backward()
    opt_actor.step()

    total_reward = np.sum([path["reward"].sum() for path in paths])
    average_reward = total_reward / time_step_batch

    writer.add_scalar("tag/reward", average_reward, iter_index)
    writer.add_scalar("tag/a_loss", a_loss.item(), iter_index)
    writer.add_scalar("tag/c_loss", v_loss.item(), iter_index)

    # average reward smoothing
    if arg.smoothing == True:
        if len(r_plot) == 0:
            r_plot.append(average_reward)
        else:
            r_plot.append(average_reward * 0.5 + r_plot[-1] * 0.5)
    else:
        r_plot.append(average_reward)
    
    # evaluate policy and record the results
    if iter_index % 20 == 0:
        tar,start = policy_evaluate(env, actor, "sto")
        tars.append(tar)
        writer.add_scalar("tag/return", tar, iter_index)
        log_trace = "Iteration: {:3d} |" \
                    "Average reward: {:7.3f} |" \
                    "Average return: {:7.3f}".format(iter_index, average_reward, tar)
        print(log_trace)
        
    if iter_index % 20 == 0:
        actor.save_parameter(log_dir, iter_index)

# ================= save result =======================
np.save(os.path.join(log_dir, "r.npy"), r_plot)
actor.save_parameter(log_dir)
np.save(os.path.join(log_dir, "tar.npy"), tars)
