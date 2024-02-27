"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 7:  plot trajectory of trained agent with error bound
"""
import lib
import gym
from lib.utils import myplot_var, smooth
from lib.idplot import self_plot_shadow
import numpy as np
import os
from lib.agent import StochasticPolicy, DeterminePolicy
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

# figure folder
fig_dir = "./figures_shadow"
os.makedirs(fig_dir, exist_ok=True)

MAX_STEP_EP = 200
RHO0 = 100
DT = 0.05
EXPECTED_V = 15 # expected vehicle speed

my_loc = "upper right"
color_dark = ["#1f77b4", "#ff7f0e"]
color_light = ["lightblue", "lightsalmon"]
lg = ["w/ BL",
      "w/o BL"]

#  =================== plot control ==================
# sample a trajectory
logdir_no  = [
    r"Results_dir/sto_sta/2021-10-13-16-22-52/actor.pth",
    r"Results_dir/sto_sta/2021-10-13-16-22-56/actor.pth",
    r"Results_dir/sto_sta/2021-10-13-16-23-14/actor.pth",
    r"Results_dir/sto_sta/2021-10-13-17-24-50/actor.pth",  # 4e-3
    r"Results_dir/sto_sta/2021-10-13-20-21-13/actor.pth",
    r"Results_dir/sto_sta/2021-10-13-20-22-09/actor.pth",
    r"Results_dir/sto_sta/2021-10-13-20-22-16/actor.pth",  # 2e-3
    r"Results_dir/sto_sta/2021-10-14-17-27-19/actor.pth",
    r"Results_dir/sto_sta/2021-10-14-17-27-31/actor.pth",
    r"Results_dir/sto_sta/2021-10-14-17-27-34/actor.pth",  # 8e-4
]
logdir_yes  = [
    r"Results_dir/sto_sta_baseline/2021-11-01-15-10-28/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-10-28/actor_1800.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-16-58/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-16-58/actor_1800.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-17-06/actor.pth",  # 6e-5
    r"Results_dir/sto_sta_baseline/2021-11-01-15-17-06/actor_1800.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-44-53/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-44-53/actor_1800.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-45-08/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-45-08/actor_1800.pth",  # 1e-4
]
'''
# small learning rate
logdir_yes  = [
    r"Results_dir/sto_sta_baseline/2021-11-01-15-10-28/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-16-09/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-16-58/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-16-38/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-15-17-06/actor.pth",  # 6e-5
    r"Results_dir/sto_sta_baseline/2021-11-01-18-44-44/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-44-53/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-45-05/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-45-08/actor.pth",
    r"Results_dir/sto_sta_baseline/2021-11-01-18-45-14/actor.pth",  # 1e-4
]
'''

def sample_trajectory(logdir, mode, F_noi=0):
    seed_everything(0)

    env = gym.make('Lanekeeping-v0')
    s = env.reset(init_state = np.array([1, 0.05, -1.5, 0.2, 0]))
    env.reset_F_fix(F_noi)

    S_DIM = env.observation_space.shape[0]  # dimension of state space
    A_DIM = env.action_space.shape[0]  # dimension of action space

    if mode == "sto":
        actor = StochasticPolicy(S_DIM, A_DIM)
    elif mode == "det":
        actor = DeterminePolicy(S_DIM, A_DIM)
    else:
        raise ValueError("Unknow mode")

    state_dict = torch.load(logdir)
    actor.load_state_dict(state_dict)

    states = [s, ]
    actions = []
    rewards = []
    time_step_ep = 0
    
    while True:
        _, a = actor.choose_action(s)
        actions.append(a)
        s, r, done, _ = env.step(a[0])
        
        states.append(s)
        rewards.append(r)
        time_step_ep += 1
        if time_step_ep >= MAX_STEP_EP:
            break

    s_buffer = np.array(states)[:-1, :]
    rho = s_buffer[:, 0]
    phi = s_buffer[:, 1]
    v_x = s_buffer[:, 2]
    v_y = s_buffer[:, 3]
    omega = s_buffer[:, 4]
    actions = np.concatenate(actions)
    tar = np.sum(rewards)
    return rho, phi, v_x, v_y, omega, actions[:,0], actions[:,1], tar



# ================== sample ======================
rho_with = []
rho_without = []
phi_with = []
phi_without = []
vx_with = []
vx_without = []
vy_with = []
vy_without = []
omega_with = []
omega_without = []
delta_with = []
delta_without = []
acc_with = []
acc_without = []
tar_with = []
tar_without = []
for i in range(10):
    rho_no, phi_no, v_x_no, v_y_no, omega_no, delta_no, acc_no, tar_no \
        = sample_trajectory(logdir_no[i], "sto")
    rho_yes, phi_yes, v_x_yes, v_y_yes, omega_yes, delta_yes, acc_yes, tar_yes \
        = sample_trajectory(logdir_yes[i], "sto")
    rho_with.append(rho_yes)
    rho_without.append(rho_no)
    phi_with.append(phi_yes * 180 / 3.14159)
    phi_without.append(phi_no * 180 / 3.14159)
    vx_with.append(v_x_yes + EXPECTED_V)
    vx_without.append(v_x_no + EXPECTED_V)
    vy_with.append(v_y_yes)
    vy_without.append(v_y_no)
    omega_with.append(omega_yes)
    omega_without.append(omega_no)
    delta_with.append(delta_yes * 20)
    delta_without.append(delta_no * 20)
    acc_with.append(acc_yes)
    acc_without.append(acc_no)
    tar_with.append(tar_yes)
    tar_without.append(tar_no)
    
# ============== plot rho error ==================
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, rho_without[0], rho_without[1], rho_without[2], rho_without[3], rho_without[4], rho_without[5], rho_without[6], rho_without[7], rho_without[8], rho_without[9]),
        (time_list, rho_with[0], rho_with[1], rho_with[2], rho_with[3], rho_with[4], rho_with[5], rho_with[6], rho_with[7], rho_with[8], rho_with[9]))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'rho.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$\rho$ [m]',
          legend=lg, yline=0,
          legend_loc=my_loc,
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ================plot phi error ============
data = ((time_list, phi_without[0], phi_without[1], phi_without[2], phi_without[3], phi_without[4], phi_without[5], phi_without[6], phi_without[7], phi_without[8], phi_without[9]),
        (time_list, phi_with[0], phi_with[1], phi_with[2], phi_with[3], phi_with[4], phi_with[5], phi_with[6], phi_with[7], phi_with[8], phi_with[9]))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'phi.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$\psi$ [deg]',
          legend=lg, yline=0,
          legend_loc="upper right",
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ============== plot v_x error ==================
data = ((time_list, vx_without[0], vx_without[1], vx_without[2], vx_without[3], vx_without[4], vx_without[5], vx_without[6], vx_without[7], vx_without[8], vx_without[9]),
        (time_list, vx_with[0], vx_with[1], vx_with[2], vx_with[3], vx_with[4], vx_with[5], vx_with[6], vx_with[7], vx_with[8], vx_with[9]))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'v_x.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$u$ [m/s]',
          legend=lg, yline=15,
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ============== plot v_y error ==================
data = ((time_list, vy_without[0], vy_without[1], vy_without[2], vy_without[3], vy_without[4], vy_without[5], vy_without[6], vy_without[7], vy_without[8], vy_without[9]),
        (time_list, vy_with[0], vy_with[1], vy_with[2], vy_with[3], vy_with[4], vy_with[5], vy_with[6], vy_with[7], vy_with[8], vy_with[9]))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'v_y.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$v$ [m/s]',
          legend=lg,
          legend_loc=my_loc,
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ============== plot omega ================
data = ((time_list, omega_without[0], omega_without[1], omega_without[2], omega_without[3], omega_without[4], omega_without[5], omega_without[6], omega_without[7], omega_without[8], omega_without[9]),
        (time_list, omega_with[0], omega_with[1], omega_with[2], omega_with[3], omega_with[4], omega_with[5], omega_with[6], omega_with[7], omega_with[8], omega_with[9]))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'omega.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$\omega$ [rad/s]',
          legend=lg,
          legend_loc=my_loc,
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ============= plot action (delta) ===============
data = ((time_list, delta_without[0], delta_without[1], delta_without[2], delta_without[3], delta_without[4], delta_without[5], delta_without[6], delta_without[7], delta_without[8], delta_without[9]),
        (time_list, smooth(delta_with[0],0.85), smooth(delta_with[1],0.85), smooth(delta_with[2],0.85), smooth(delta_with[3],0.85), smooth(delta_with[4],0.85), smooth(delta_with[5],0.85), smooth(delta_with[6],0.85), smooth(delta_with[7],0.85), smooth(delta_with[8],0.85), smooth(delta_with[9],0.85)))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'delta.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$\delta$ [deg]',
          legend=lg, yline=0,
          legend_loc=my_loc,
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

# ============= plot action (acc) ===============
data = ((time_list, acc_without[0], acc_without[1], acc_without[2], acc_without[3], acc_without[4], acc_without[5], acc_without[6], acc_without[7], acc_without[8], acc_without[9]),
        (time_list, smooth(acc_with[0],0.85), smooth(acc_with[1],0.85), smooth(acc_with[2],0.85), smooth(acc_with[3],0.85), smooth(acc_with[4],0.85), smooth(acc_with[5],0.85), smooth(acc_with[6],0.85), smooth(acc_with[7],0.85), smooth(acc_with[8],0.85), smooth(acc_with[9],0.85)))
data = (data[1],data[0])
self_plot_shadow(data,
          os.path.join(fig_dir, 'acc.png'),
          xlabel=r"$t$ [s]",
          ylabel=r'$a_{x}$ $\mathregular{[m/s^2]}$',
          legend=lg, yline=0,
          legend_loc=my_loc,
          color_dark=color_dark,
          color_light=color_light,
          display=False,
          )

print("tar_miu (without baseline) = ", np.mean(tar_without))
print("tar_std (without baseline) = ", np.std(tar_without))
print("tar_miu (with baseline) = ", np.mean(tar_with))
print("tar_std (with baseline) = ", np.std(tar_with))