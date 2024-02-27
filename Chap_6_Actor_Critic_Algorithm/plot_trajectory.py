"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 6:  plot trajectory of trained agent
"""
import lib
import gym
from lib.utils import myplot_var, smooth
from lib.idplot import self_plot
import numpy as np
import os
from lib.agent import StochasticPolicy, DeterminePolicy
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

# figure folder
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)

MAX_STEP_EP = 200
RHO0 = 100
DT = 0.05
EXPECTED_V = 15 # expected vehicle speed

my_loc = "upper right"
colorl = ["#1f77b4", "#ff7f0e", "#2ca02c"]
lg = ["Sto-V",
      "Det-Q",
      "Sto-Q", ]

#  =================== plot control ==================
# sample a trajectory
logdir0 = r"Results_dir/sto_sta/"
logdir1 = r"Results_dir/det_act/"
logdir2 = r"Results_dir/sto_act/"

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
        raise ValueError("Unknown mode")

    state_dict = torch.load(os.path.join(logdir, "actor.pth"))
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
    ret = np.sum(rewards)
    return rho, phi, v_x, v_y, omega, actions, ret


# ================== sample ======================
rho_ss, phi_ss, v_x_ss, v_y_ss, omega_ss, actions_ss, ret_ss \
    = sample_trajectory(logdir0, "sto")
rho_da, phi_da, v_x_da, v_y_da, omega_da, actions_da, ret_da \
    = sample_trajectory(logdir1, "det")
rho_sa, phi_sa, v_x_sa, v_y_sa, omega_sa, actions_sa, ret_sa \
    = sample_trajectory(logdir2, "sto")

# ============== plot rho error ==================
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, rho_ss),
        (time_list, rho_da),
        (time_list, rho_sa)
        )
self_plot(data,
       os.path.join(fig_dir, "rho.png"),
       xlabel=r"$t$ [s]",
       ylabel=r'$\rho$ [m]',
       legend=lg, yline=0,
       legend_loc="upper right",
       color_list=colorl,
       display=False)

# ================plot phi error ============
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, smooth(phi_ss * 180 / 3.14159, 1)),
        (time_list, smooth(phi_da * 180 / 3.14159, 1)),
        (time_list, smooth(phi_sa * 180 / 3.14159, 1))
        )
self_plot(data,
       os.path.join(fig_dir, "phi.png"),
       xlabel=r"$t$ [s]",
       legend_loc=my_loc,
       ylabel=r'$\psi$ [deg]', color_list=colorl,
       legend=lg, yline=0,
       display=False)

# ============== plot v_x error ==================
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, v_x_ss + EXPECTED_V),
        (time_list, v_x_da + EXPECTED_V),
        (time_list, v_x_sa + EXPECTED_V)
        )
self_plot(data,
       os.path.join(fig_dir, "v_x.png"),
       xlabel=r"$t$ [s]",
       ylabel=r'$u$ [m/s]',
       legend=lg, yline=15,
       legend_loc="lower right",
       color_list=colorl,
       display=False)

# ============== plot v_y error ==================
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, smooth(v_y_ss, 1)),
        (time_list, smooth(v_y_da, 1)),
        (time_list, smooth(v_y_sa, 1))
        )
self_plot(data,
       os.path.join(fig_dir, "v_y.png"),
       xlabel=r"$t$ [s]",
       ylabel=r'$v$ [m/s]',
       legend=lg,
       legend_loc="upper right",
       color_list=colorl,
       display=False)

# ============== plot omega ================
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, smooth(omega_ss, 1)),
        (time_list, smooth(omega_da, 1)),
        (time_list, smooth(omega_sa, 1))
        )
self_plot(data,
       os.path.join(fig_dir, "omega.png"),
       xlabel=r"$t$ [s]",
       legend=lg, legend_loc=my_loc,
       ylabel=r'$\omega$ [rad/s]', color_list=colorl, ytick=[0.8, 0.6, 0.4, 0.2, 0, -0.2],
       display=False)

# ============= plot action (delta) ===============
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, smooth(actions_ss[:, 0] * 20, 1)),
        (time_list, smooth(actions_da[:, 0] * 20, 1)),
        (time_list, smooth(actions_sa[:, 0] * 20, 1))
        )
self_plot(data,
       os.path.join(fig_dir, "delta.png"),
       xlabel=r"$t$ [s]",
       legend=lg, legend_loc=my_loc,
       ylabel=r'$\delta$ [deg]', color_list=colorl,yline=0,
       display=False)

# ============= plot action (acc) ===============
time_list = [i * DT for i in range(MAX_STEP_EP)]
data = ((time_list, smooth(actions_ss[:, 1], 1)),
        (time_list, smooth(actions_da[:, 1], 1)),
        (time_list, smooth(actions_sa[:, 1], 1))
        )
self_plot(data,
       os.path.join(fig_dir, "acc.png"),
       xlabel=r"$t$ [s]",
       legend=lg, legend_loc=my_loc,
       ylabel=r'$a_{x}$ $\mathregular{[m/s^2]}$', color_list=colorl, yline=0,
       display=False)
