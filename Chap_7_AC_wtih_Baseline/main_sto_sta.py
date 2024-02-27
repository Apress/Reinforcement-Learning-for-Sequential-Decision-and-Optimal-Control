"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 7:  RL example for lane keeping problem in a circle road
             Actor-Critic algorithm with stochastic policy and state value function
             with baseline comparison

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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything

from lib import StochasticPolicy, StateValue
from lib import policy_evaluate
from lib import self_plot
from lib.statistics import RunningMeanStd

# ============= Setting hyper-parameters ===============
parse = argparse.ArgumentParser(description="Training parameters")
parse.add_argument("--baseline", action="store_true", help="add baseline", default=True)
parse.add_argument("--norm", action="store_true", help="normalization of bootstrap", default=True)
parse.add_argument("--lr_a", type=float, default=5e-4, help="learning rate of actor")  # 5e-3
parse.add_argument("--lr_c", type=float, default=5e-4, help="learning rate of critic")
parse.add_argument("--gamma", type=float, default=0.97, help="parameter gamma")  # 0.97 default
parse.add_argument("--iter_num", type=int, default=2000, help="total number of training iteration")
parse.add_argument("--batch_size", type=int, default=256, help="minimum samples in a update batch")
parse.add_argument("--step_limit", type=int, default=128, help="maximum step the agent can run in the environment")
parse.add_argument("--seed", type=int, default=0, help="random seed")
parse.add_argument("--mb_num", type=int, default=4, help="mini-batch number")
parse.add_argument("--smoothing", action="store_true")
parse.add_argument("--actor_lr_schedule", type=list, default=[5e-4, parse.parse_args().iter_num, 1e-6],)
parse.add_argument("--critic_lr_schedule", type=list, default=[5e-4, parse.parse_args().iter_num, 1e-6],)
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
ret_rms = RunningMeanStd()
EPS = 1e-8
S_DIM = env.observation_space.shape[0]  # dimension of state space
A_DIM = env.action_space.shape[0]  # dimension of action space

actor = StochasticPolicy(S_DIM, A_DIM)
critic = StateValue(S_DIM, 1)

opt_actor = torch.optim.Adam(actor.parameters(), arg.lr_a)
opt_critic = torch.optim.Adam(critic.parameters(), arg.lr_c)

actor_init_lr, actor_decay_steps, actor_end_lr = arg.actor_lr_schedule
actor_lr_schedule = LambdaLR(opt_actor, lr_lambda=lambda epoch: (1. - actor_end_lr / actor_init_lr) *
                                                                (1. - min(epoch, actor_decay_steps) / actor_decay_steps) +
                                                                actor_end_lr / actor_init_lr)
critic_init_lr, critic_decay_steps, critic_end_lr = arg.critic_lr_schedule
critic_lr_schedule = LambdaLR(opt_actor, lr_lambda=lambda epoch: (1. - critic_end_lr / critic_init_lr) *
                                                                 (1. - min(epoch, critic_decay_steps) / critic_decay_steps) +
                                                                 critic_end_lr / critic_init_lr)



# logdir = './Results_dir/sto_sta/2022-05-20-12-40-56/actor_1800.pth'
# state_dict = torch.load(logdir)
# actor.load_state_dict(state_dict)

# ====================== Training ========================
r_plot = []
tars = []
for iter_index in range(arg.iter_num):
    paths = []

    def sample():
        # roll out (s, a, s', r, done)
        time_step_batch = 0
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

            ret_rms.update(np.array(rewards))  # 更新标准差

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

        return state_tensor, action_tensor, v_predict_tensor, state_next_tensor, reward_tensor, terminal_tensor, time_step_batch

    def sample_mb(mb_num):
        state_tensor_mb_list = []
        action_tensor_mb_list = []
        v_predict_tensor_mb_list = []
        state_next_tensor_mb_list = []
        reward_tensor_mb_list = []
        terminal_tensor_mb_list = []
        time_step_batch_mb_list = []
        for i in range(mb_num):
            state_tensor, action_tensor, v_predict_tensor, state_next_tensor, reward_tensor, terminal_tensor, time_step_batch = sample()
            state_tensor_mb_list += [state_tensor]
            action_tensor_mb_list += [action_tensor]
            v_predict_tensor_mb_list += [v_predict_tensor]
            state_next_tensor_mb_list += [state_next_tensor]
            reward_tensor_mb_list += [reward_tensor]
            terminal_tensor_mb_list += [terminal_tensor]
            time_step_batch_mb_list += [time_step_batch]

        return state_tensor_mb_list, action_tensor_mb_list, v_predict_tensor_mb_list, state_next_tensor_mb_list, reward_tensor_mb_list, terminal_tensor_mb_list, time_step_batch_mb_list

    def update(state_tensor, action_tensor, v_predict_tensor, state_next_tensor, reward_tensor, terminal_tensor):
        # Critic loss & update
        for _ in range(10):
            v_next_tensor = critic(state_next_tensor)
            v_target_tensor = reward_tensor + arg.gamma * (1 - terminal_tensor) * v_next_tensor
            v_predict_tensor = critic(state_tensor.detach())
            v_loss = F.mse_loss(v_predict_tensor, v_target_tensor.detach())
            opt_critic.zero_grad()
            v_loss.backward()
            opt_critic.step()
        # print('critic_lr_schedule', critic_lr_schedule.get_lr())

        # Actor loss & update
        if arg.baseline is True:
            bootstrap = v_target_tensor - critic(state_tensor)  # 要不要 * np.sqrt(ret_rms.var + EPS)
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
        # print('actor_lr_schedule', actor_lr_schedule.get_lr())

    state_mb_list, action_mb_list, v_predict_mb_list, state_next_mb_list, reward_mb_list, terminal_mb_list, time_step_batch_mb_list = sample_mb(arg.mb_num)

    for i in range(arg.mb_num):
        state_tensor = state_mb_list[i]
        action_tensor = action_mb_list[i]
        v_predict_tensor = v_predict_mb_list[i]
        state_next_tensor = state_next_mb_list[i]
        reward_tensor = reward_mb_list[i]
        terminal_tensor = terminal_mb_list[i]
        time_step_batch = time_step_batch_mb_list[i]

        update(state_tensor, action_tensor, v_predict_tensor, state_next_tensor, reward_tensor, terminal_tensor)
    actor_lr_schedule.step()
    critic_lr_schedule.step()

    total_reward = np.sum([path["reward"].sum() for path in paths])
    average_reward = total_reward / time_step_batch

    writer.add_scalar("tag/reward", average_reward, iter_index)
    # writer.add_scalar("tag/a_loss", a_loss.item(), iter_index)
    # writer.add_scalar("tag/c_loss", v_loss.item(), iter_index)

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
                    "Average return: {:7.3f} |"\
                    "actor_lr: {:7.3e} |"\
                    "critic_lr: {:7.3e}".format(iter_index, average_reward, tar, actor_lr_schedule.get_lr()[0], critic_lr_schedule.get_lr()[0])
        print(log_trace)
        
    if iter_index % 20 == 0:
        actor.save_parameter(log_dir, iter_index)

# ================= save result =======================
np.save(os.path.join(log_dir, "r.npy"), r_plot)
actor.save_parameter(log_dir)
np.save(os.path.join(log_dir, "tar.npy"), tars)
