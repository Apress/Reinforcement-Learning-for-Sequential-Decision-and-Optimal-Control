"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

"""

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


default_cfg = AttrDict()

default_cfg.fig_size = (8.5, 6.5)
default_cfg.fig_size_squre = (6.5, 6.5)
default_cfg.dpi = 300
default_cfg.pad = 0.2
default_cfg.figsize_scalar = 1
default_cfg.tick_size = 8
default_cfg.linewidth = 2
default_cfg.tick_label_font = 'Times New Roman'
default_cfg.legend_font = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
default_cfg.label_font = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}