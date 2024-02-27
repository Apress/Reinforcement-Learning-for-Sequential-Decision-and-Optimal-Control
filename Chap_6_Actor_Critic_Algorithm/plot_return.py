"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 6:  plot the return of different algorithms
"""
from lib.idplot import self_plot, self_plot_shadow
from lib.utils import smooth, aggre
import os
import sys
import numpy as np

MAX_ITERATION = 2000
ITER_STEP = 5

save_path = './figures'
os.makedirs(save_path, exist_ok=True)

xlabel = 'Iteration Steps'
ylabel = 'Total Average Return'
ylim = [0, 4000]
color_dark = ["#1f77b4", "#ff7f0e", "#2ca02c"]
color_light = ["lightblue", "lightsalmon", "lightgreen"]

# ========== deterministic policy and action value ==========
data11 = np.load('./Results_dir/det_act/2021-10-11-23-28/tar.npy')
data12 = np.load('./Results_dir/det_act/2021-10-11-23-42/tar.npy')
data13 = np.load('./Results_dir/det_act/2021-10-11-23-47/tar.npy')
data14 = np.load('./Results_dir/det_act/2021-10-12-00-23/tar.npy')
data15 = np.load('./Results_dir/det_act/2021-10-12-00-31/tar.npy')

data21 = np.load('./Results_dir/det_act/2021-10-12-00-36-46/tar.npy')
data22 = np.load('./Results_dir/det_act/2021-10-12-00-37-03/tar.npy')
data23 = np.load('./Results_dir/det_act/2021-10-12-00-37-08/tar.npy')
data24 = np.load('./Results_dir/det_act/2021-10-12-00-37-18/tar.npy')
data25 = np.load('./Results_dir/det_act/2021-10-12-00-37-25/tar.npy')

data31 = np.load('./Results_dir/det_act/2021-10-12-10-41-58/tar.npy')
data32 = np.load('./Results_dir/det_act/2021-10-12-10-42-02/tar.npy')
data33 = np.load('./Results_dir/det_act/2021-10-12-10-42-11/tar.npy')
data34 = np.load('./Results_dir/det_act/2021-10-12-10-42-44/tar.npy')
data35 = np.load('./Results_dir/det_act/2021-10-12-10-42-48/tar.npy')

time_list = np.arange(0, MAX_ITERATION, 20)
data = ((time_list, smooth(data11,1), smooth(data12,1), smooth(data13,1), smooth(data14,1), smooth(data15,1)), 
        (time_list, smooth(data21,1), smooth(data22,1), smooth(data23,1), smooth(data24,1), smooth(data25,1)),
        (time_list, aggre(smooth(data31,1),4), aggre(smooth(data32,1),4), aggre(smooth(data33,1),4), aggre(smooth(data34,1),4), aggre(smooth(data35,1),4)))
self_plot_shadow(data,
          os.path.join(save_path, 'ac-det_act.png'),
          xlabel=xlabel,
          ylabel=ylabel,
          legend=[r'$\alpha=\beta=$6e-4', r'$\alpha=\beta=$1e-4', r'$\alpha=\beta=$8e-5'],
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          ylim=ylim,
          yline=3900,
          display=False)

# ========== stochastic policy and action value ==========
data11 = np.load('./Results_dir/sto_act/2021-10-12-15-16-43/tar.npy')
data12 = np.load('./Results_dir/sto_act/2021-10-12-15-16-48/tar.npy')
data13 = np.load('./Results_dir/sto_act/2021-10-12-15-17-13/tar.npy')
data14 = np.load('./Results_dir/sto_act/2021-10-12-15-16-52/tar.npy')
data15 = np.load('./Results_dir/sto_act/2021-10-12-15-17-01/tar.npy')

data21 = np.load('./Results_dir/sto_act/2021-10-12-16-46-34/tar.npy')
data22 = np.load('./Results_dir/sto_act/2021-10-12-16-46-49/tar.npy')
data23 = np.load('./Results_dir/sto_act/2021-10-12-16-47-06/tar.npy')
data24 = np.load('./Results_dir/sto_act/2021-10-12-16-46-56/tar.npy')
data25 = np.load('./Results_dir/sto_act/2021-10-12-16-47-10/tar.npy')

data31 = np.load('./Results_dir/sto_act/2021-10-13-12-05-59/tar.npy')
data32 = np.load('./Results_dir/sto_act/2021-10-13-12-06-04/tar.npy')
data33 = np.load('./Results_dir/sto_act/2021-10-13-12-06-07/tar.npy')
data34 = np.load('./Results_dir/sto_act/2021-10-13-12-05-53/tar.npy')
data35 = np.load('./Results_dir/sto_act/2021-10-13-12-06-16/tar.npy')

time_list = np.arange(0, MAX_ITERATION, ITER_STEP)
data = ((aggre(time_list,4), aggre(smooth(data11,1),4), aggre(smooth(data12,1),4), aggre(smooth(data13,1),4), aggre(smooth(data14,1),4), aggre(smooth(data15,1),4)), 
        (aggre(time_list,4), aggre(smooth(data21,1),4), aggre(smooth(data22,1),4), aggre(smooth(data23,1),4), aggre(smooth(data24,1),4), aggre(smooth(data25,1),4)),
        (aggre(time_list,4), aggre(smooth(data31,1),4), aggre(smooth(data32,1),4), aggre(smooth(data33,1),4), aggre(smooth(data34,1),4), aggre(smooth(data35,1),4)))
self_plot_shadow(data,
          os.path.join(save_path, 'ac-sto_act.png'),
          xlabel=xlabel,
          ylabel=ylabel,
          legend=[r'$\alpha=\beta=$6e-4', r'$\alpha=\beta=$1e-4', r'$\alpha=\beta=$6e-5'],
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          ylim=ylim,
          yline=3900,
          display=False)

# ========== stochastic policy and state value ==========
data11 = np.load('./Results_dir/sto_sta/2021-10-31-20-50-17/tar.npy')
data12 = np.load('./Results_dir/sto_sta/2021-10-13-16-22-56/tar.npy')
data13 = np.load('./Results_dir/sto_sta/2021-10-13-17-24-50/tar.npy')
data14 = np.load('./Results_dir/sto_sta/2021-10-13-16-23-14/tar.npy')
data15 = np.load('./Results_dir/sto_sta/2021-10-31-16-34-40/tar.npy')

data21 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-15/tar.npy')
data22 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-19/tar.npy')
data23 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-31/tar.npy')
data24 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-34/tar.npy')
data25 = np.load('./Results_dir/sto_sta/2021-10-14-17-51-01/tar.npy')

data31 = np.load('./Results_dir/sto_sta/2021-10-13-20-21-13/tar.npy')
data32 = np.load('./Results_dir/sto_sta/2021-10-13-20-21-56/tar.npy')
data33 = np.load('./Results_dir/sto_sta/2021-10-13-20-22-30/tar.npy')
data34 = np.load('./Results_dir/sto_sta/2021-10-13-20-22-16/tar.npy')
data35 = np.load('./Results_dir/sto_sta/2021-10-29-17-34-25/tar.npy')

time_list = np.arange(0, MAX_ITERATION, ITER_STEP)
data = ((aggre(time_list,4), aggre(smooth(data11,1),1), aggre(smooth(data12,1),1), aggre(smooth(data13,1),1), aggre(smooth(data14,1),1), aggre(smooth(data15,1),1)),
        (aggre(time_list,4), aggre(smooth(data21,1),1), aggre(smooth(data22,1),1), aggre(smooth(data23,1),1), aggre(smooth(data24,1),1), aggre(smooth(data25,1),1)),
        (aggre(time_list,4), aggre(smooth(data31,1),1), aggre(smooth(data32,1),1), aggre(smooth(data33,1),1), aggre(smooth(data34,1),1), aggre(smooth(data35,1),1)))
self_plot_shadow(data,
          os.path.join(save_path, 'ac-sto_sta.png'),
          xlabel=xlabel,
          ylabel=ylabel,
          legend=[r'$\alpha=\beta=$4e-3', r'$\alpha=\beta=$2e-3', r'$\alpha=\beta=$8e-4'],
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          ylim=ylim,
          yline=3900,
          display=False)
