"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang

Description: Chapter 9: Feasibility analysis with emergency braking control
             Model Predictive Control with finite horizon
             Draw state trajectories of MPC
             "ED" = Exponentially decaying constraint
             "PW" = Pointwise constraint
"""

import os

from Convergence import Convergence
import matplotlib.pyplot as  plt
import numpy as np
from Config import cfg_default, cfg_matplotlib
from matplotlib import rcParams

rcParams.update({'mathtext.fontset':'stix'})

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

# Select your own settings for constraints
# Format = (Horizon length, True="ED"/False="PW", Lambda (if using ED))

# Conditon A: With less initial states
cfgs_run = [
        # (5, False, 1),

        (2, True, 0.02),
        (2, True, 0.10),
        (2, True, 0.18),
        (2, True, 0.25),
        (2, False, 1),
        (4, False, 1),
        (6, False, 1),
        (20, False, 1),
    ]
start_points = [   # (d, u)
        (8, 11),
        (6, 3),
        (7, 8),
        (10, 5),
        (2, 1),
    ]

# Condition B: With more initial states
# cfgs_run = [
#         (5, False, 1),
#     ]
# start_points = [   # (d, u)
#         (8, 11),
#         (6, 3),
#         (4, 12),
#         (7, 8),
#         (10, 5),
#         (2, 1),
#         (6, 14),
#     ]



def run():
    for Np, CBF, cbf_para in cfgs_run:
        cfg = cfg_default
        cfg["flag"] = True
        cfg["Np"] = Np
        cfg["CBF"] = CBF
        cfg["cbf_para"] = cbf_para
        convergence = Convergence()

        for x in start_points:
            convergence.examine(x, cfg)

def plot():

    for idx, cfg in enumerate(cfgs_run):
        plt.figure(figsize=cfg_matplotlib["fig_size"], dpi=cfg_matplotlib["dpi"])
        ax = plt.gca()

        # plot analytical boundary
        d_red = np.array([0.5 * i for i in range(21)])
        u_red = np.sqrt(2 * 10 * d_red)
        plt.plot(d_red, u_red, "--r")
        ax.plot([0.0, 0.0],list(ax.get_ylim()), color='grey', linestyle='--', label='Ref', zorder=0)
        # plot trajectory
        for idj, sp in enumerate(start_points):
            traj = np.load("./data/traj_{N}_{cbf}_{d}_{u}_{alpha}.npy".format(
                    N=cfg[0], 
                    cbf=str(cfg[1]), d=sp[0], u=sp[1],
                    alpha=cfg[2]
                    ))
            plt.plot(traj[:, 0], traj[:, 1], linestyle='--', marker="o",ms=4,
                     markerfacecolor='white', zorder=0, color=color_list[idj])
            
            if traj[-1, 0] < 0 and traj[-1, 1] >= 0.01:
                plt.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=50, color=color_list[idj])
            plt.scatter(traj[0, 0], traj[0, 1], marker="o", color=color_list[idj])

        method_name = "ED" if cfg[1] else "PW"
        print("======= N = {N}, Constr = {cbf} ========".format(
                    N=cfg[0], 
                    cbf=str(method_name),
                    ))
        plt.xlabel("$d$ [m]", cfg_matplotlib["label_font"])
        plt.ylabel("$v$ [m/s]", cfg_matplotlib["label_font"])
        
        title = "Pointwise constraint: $N$={}".format(cfg[0]) \
            if not cfg[1] else "Exponentially decaying constraint: $\lambda={}$".format(cfg[2])
        plt.title(title, cfg_matplotlib["label_font"], fontsize=14)
        plt.tick_params(labelsize=cfg_matplotlib["tick_size"])
        
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(cfg_matplotlib["tick_label_font"]) for label in labels]
        plt.tight_layout(pad=cfg_matplotlib["pad"])
        os.makedirs("./figure", exist_ok=True)

        # Plot figures
        if cfg[1]:
            plt.savefig("./figure/traj_{}_N{}_{}.png".format("ED", cfg[0], cfg[2]))
        else:
            plt.savefig("./figure/traj_{}_N{}.png".format("PW", cfg[0]))
if __name__ == '__main__':
    run()
    plot()