"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang

Description: Chapter 9:  OCP example for emergency braking control
             Hyper-parameter setup
"""

from __future__ import print_function


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


cfg_default = dict(
    DYNAMICS_DIM=2,
    ACTION_DIM=1,
    Ts=0.1,  # control signal period
    Np=2,  # predict horizon
    U_LOWER=-10,
    d_safe=0,
    CBF=False,
    cbf_para=0.01,
    flag=False,  # save trajectory
)

cfg_matplotlib = dict()

cfg_matplotlib["fig_size"] = (4, 4)
cfg_matplotlib["dpi"] = 300
cfg_matplotlib["pad"] = 0.5

cfg_matplotlib["tick_size"] = 10
cfg_matplotlib["tick_label_font"] = "Times New Roman"
cfg_matplotlib["legend_font"] = {
    "family": "Times New Roman",
    "size": "10",
    "weight": "normal",
}
cfg_matplotlib["label_font"] = {
    "family": "Times New Roman",
    "size": "14",
    "weight": "normal",
}