"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang

Description: Chapter 9: Feasibility analysis with emergency braking control
             Model Predictive Control with finite horizon
             Draw feasible & infeasible regions
             "ED" = Exponentially decaying constraint
             "PW" = Pointwise constraint
"""

from Convergence import Convergence
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from Config import cfg_default, cfg_matplotlib
from matplotlib import rcParams
from multiprocessing import Process
rcParams.update({'mathtext.fontset':'stix'})

# Select your own settings for constraints
# Format = (Horizon length, True="ED"/False="PW", Lambda (if using ED))
cfgs_run = [
    (2, True, 0.02),
    (2, True, 0.1),
    (2, True, 0.18),
    (2, True, 0.25),
    # (2, False, 1),
    # (4, False, 1),
    # (6, False, 1),
    # (20, False, 1),
]

def single_run(cfg):
    d_range = 2 * 10
    u_range = 2 * 14
    convergence = Convergence()
    stop1 = 0
    history = []
    for i in range(d_range):
        for j in range(u_range):
            d = 0.5 * i
            u = 0.5 * j
            crash, init_feasibility = convergence.examine([d, u], cfg)
            if crash == 0:
                stop1 += 1
            history.append([d, u, crash, init_feasibility])

    name = './data/history_{}_{}_{}.txt'.format(str(cfg["CBF"]), cfg["Np"], cfg["cbf_para"])
    np.savetxt(name, np.array(history))
    return history

def plot_figure(data, cfg):
    plt.figure(figsize=cfg_matplotlib["fig_size"], dpi=cfg_matplotlib["dpi"])
    ax = plt.gca()
    d_red = np.array([0.5 * i for i in range(21)])
    u_red = np.sqrt(2 * 10 * d_red)
    plt.plot(d_red, u_red, "--r")

    Grey_set = []
    Green_set = []
    Gold_set = []
    for data_single_point in data:
        d, u, crash, init_feasibility = data_single_point[0], \
                                        data_single_point[1], \
                                        data_single_point[2], \
                                        data_single_point[3]
        if crash == 0:
            Green_set.append(np.array([d, u]))
        elif init_feasibility == 0:
            Gold_set.append(np.array([d, u]))
        else:
            Grey_set.append(np.array([d, u]))

    Green_set = np.array(Green_set)

    plt.scatter(Green_set[:, 0], Green_set[:, 1], marker='s', color='limegreen', label="Endlessly Feasible Region")

    if len(Gold_set) > 0:
        Gold_set = np.array(Gold_set)
        plt.scatter(Gold_set[:, 0], Gold_set[:, 1], marker='x', color='gold', label='Initially Feasible Region')

    if len(Grey_set) > 0:
        Grey_set = np.array(Grey_set)
        plt.scatter(Grey_set[:, 0], Grey_set[:, 1], marker='x', color='grey', label='Infeasible Region')
    plt.xlabel('$d$ [m]', cfg_matplotlib["label_font"])
    plt.ylabel('$v$ [m/s]', cfg_matplotlib["label_font"])

    if cfg["CBF"]:
        plt.title("Exponentially decaying constraint: $\lambda={:.2f}$".format(cfg["cbf_para"]), cfg_matplotlib["label_font"])
    else:
        plt.title("Pointwise constraint: $N={}$".format(cfg["Np"]), cfg_matplotlib["label_font"])

    plt.legend(loc='lower right', prop=cfg_matplotlib["legend_font"])
    plt.tick_params(labelsize=cfg_matplotlib["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(cfg_matplotlib["tick_label_font"]) for label in labels]
    plt.tight_layout(pad=cfg_matplotlib["pad"])
    os.makedirs("./figure", exist_ok=True)

    # Save figures
    if cfg["CBF"]:
        plt.savefig("./figure/area_{}_N{}_{}.png".format("ED", cfg["Np"], cfg["cbf_para"]))
    else:
        plt.savefig("./figure/area_{}_N{}.png".format("PW", cfg["Np"]))

def run(cfg):
    method_name = "ED" if cfg["CBF"] else "PW"
    print("plot N = ", cfg["Np"], "Constr: ", method_name)
    hist = single_run(cfg)
    plot_figure(hist, cfg)
    print("Finish N = ", cfg["Np"], "Constr: ", method_name)

def main_async(cfg_list):
    start_time =time.time()
    procs = []

    for Np, CBF, cbf_para in cfg_list:
        cfg = cfg_default
        cfg["Np"] = Np
        cfg["CBF"] = CBF
        cfg["cbf_para"] = cbf_para
        procs.append(Process(target=run, args=(copy.deepcopy(cfg),)))

    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("total time: ", time.time() - start_time)

def main(cfg_list):
    for Np, CBF, cbf_para in cfg_list:
        cfg = cfg_default
        cfg["Np"] = Np
        cfg["CBF"] = CBF
        cfg["cbf_para"] = cbf_para
        hist = single_run(cfg)
        plot_figure(hist, cfg)

def plot_only(cfg_list):
    for Np, CBF, cbf_para in cfg_list:
        cfg = cfg_default
        cfg["Np"] = Np
        cfg["CBF"] = CBF
        hist = np.loadtxt('./data/history_{}_{}_{}.txt'.format(str(CBF), Np, cbf_para))
        plot_figure(hist, cfg)

if __name__ == '__main__':

    # plot_only(cfgs_run)
    # main(cfgs_run)
    main_async(cfgs_run)
