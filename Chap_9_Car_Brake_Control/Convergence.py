"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang

Description: Chapter 9:  OCP example for emergency braking control
             Feasibility examination of each state point

"""

import os
import pprint
from Solver import Solver
import numpy as np
import copy
import matplotlib.pyplot as plt

class Convergence():
    def examine(self, x, cfg):
        d0, u0 = copy.deepcopy(x)
        crash = 0
        x_traj = [x]
        u_traj = []
        solver = Solver(cfg)
        for i in range(200):
            state, control = solver.mpcSolver(copy.deepcopy(x))
            if i == 0:
                init_feasiblity = check_init_feasible([d0, u0], control, cfg)
            control = control.tolist()
            u = control[0]
            x_next = [x[0] - cfg["Ts"] * x[1], x[1] + cfg["Ts"] * u[0]]
            x = x_next
            x_traj.append(x)
            u_traj.append(u)
            if (x[0] <= 0 and x[1] > 0.01) or (d0 == 0 and u0 > 0):
                crash = 1
                if (d0 == 0 and u0 > 0): init_feasiblity = 1
                break

            if x[1] <= 0.01:
                break

        if cfg["flag"]:
            os.makedirs("./data", exist_ok=True)
            trajectory_np = np.array(x_traj)

            np.save(
                "./data/traj_{N}_{cbf}_{d}_{u}_{alpha}.npy".format(
                    N=cfg["Np"],
                    cbf=str(cfg["CBF"]), d=d0, u=u0,
                    alpha=cfg["cbf_para"]
                    ),
                    trajectory_np
                    )
        return crash, init_feasiblity


def check_init_feasible(state, control, cfg):
    init_state = state
    a = int(0)
    # if state[0] =
    state_buffer = [state]
    control_buffer = []

    # print(cfg)
    if not cfg["CBF"]:
        # PW
        for u in control:
            state = ff(state, u, cfg["Ts"])
            state_buffer.append(state)
            control_buffer.append(u)
            if state[0] <= 0 and state[1] > 0.01:
                a = int(1)
    else:
        # print(":", control)
        for u in control:
            state = ff(state, u, cfg["Ts"])
            state_buffer.append(state)
            control_buffer.append(u)
        hpp = state_buffer[2][0]
        h = state_buffer[0][0]
        if hpp < (1 - cfg["cbf_para"]) * (1 - cfg["cbf_para"]) * h and h > 0:
            a = int(1)

    return a


def ff(x, cont, dt):
    x_next = [x[0] - dt * x[1], x[1] + dt * cont[0]]
    return x_next



