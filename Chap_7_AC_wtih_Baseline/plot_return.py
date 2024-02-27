"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang & Xujie Song

Description: Chapter 7:  plot the return of different algorithms
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
legend_without = [r'$\alpha=\beta=$4e-3', r'$\alpha=\beta=$2e-3', r'$\alpha=\beta=$8e-4']
legend_with = [r'$\alpha=\beta=$2e-3', r'$\alpha=\beta=$1e-4', r'$\alpha=\beta=$6e-5']

# ========== stochastic policy and state value without baseline ==========
data10 = np.load('./Results_dir/sto_sta/2021-10-13-16-22-52/tar.npy')
data11 = np.load('./Results_dir/sto_sta/2021-10-13-16-22-56/tar.npy')
data12 = np.load('./Results_dir/sto_sta/2021-10-13-17-24-50/tar.npy')
data13 = np.load('./Results_dir/sto_sta/2021-10-13-16-23-14/tar.npy')
data14 = np.load('./Results_dir/sto_sta/2021-10-31-21-21-47/tar.npy')
data15 = np.load('./Results_dir/sto_sta/2021-10-31-16-34-40/tar.npy')
data16 = np.load('./Results_dir/sto_sta/2021-10-31-16-34-45/tar.npy')
data17 = np.load('./Results_dir/sto_sta/2021-10-31-19-30-48/tar.npy')
data18 = np.load('./Results_dir/sto_sta/2021-10-31-19-59-21/tar.npy')
data19 = np.load('./Results_dir/sto_sta/2021-10-31-20-50-17/tar.npy')

data20 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-15/tar.npy')
data21 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-19/tar.npy')
data22 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-31/tar.npy')
data23 = np.load('./Results_dir/sto_sta/2021-10-14-17-27-34/tar.npy')
data24 = np.load('./Results_dir/sto_sta/2021-10-14-17-51-01/tar.npy')
data25 = np.load('./Results_dir/sto_sta/2021-10-29-17-32-33/tar.npy')
data26 = np.load('./Results_dir/sto_sta/2021-10-29-17-32-51/tar.npy')
data27 = np.load('./Results_dir/sto_sta/2021-10-29-17-32-59/tar.npy')
data28 = np.load('./Results_dir/sto_sta/2021-10-29-17-33-35/tar.npy')
data29 = np.load('./Results_dir/sto_sta/2021-10-29-17-33-46/tar.npy')

data30 = np.load('./Results_dir/sto_sta/2021-10-13-20-21-13/tar.npy')
data31 = np.load('./Results_dir/sto_sta/2021-10-13-20-21-56/tar.npy')
data32 = np.load('./Results_dir/sto_sta/2021-10-13-20-22-30/tar.npy')
data33 = np.load('./Results_dir/sto_sta/2021-10-13-20-22-16/tar.npy')
data34 = np.load('./Results_dir/sto_sta/2021-10-13-20-22-30/tar.npy')
data35 = np.load('./Results_dir/sto_sta/2021-10-29-17-34-25/tar.npy')
data36 = np.load('./Results_dir/sto_sta/2021-10-29-17-35-02/tar.npy')
data37 = np.load('./Results_dir/sto_sta/2021-10-29-17-35-15/tar.npy')
data38 = np.load('./Results_dir/sto_sta/2021-10-29-17-35-22/tar.npy')
data39 = np.load('./Results_dir/sto_sta/2021-10-29-17-35-52/tar.npy')

time_list = np.arange(0, MAX_ITERATION, ITER_STEP)
data = ((aggre(time_list,4), aggre(smooth(data10,1),1), aggre(smooth(data11,1),1), aggre(smooth(data12,1),1), aggre(smooth(data13,1),1), aggre(smooth(data14,1),1), aggre(smooth(data15,1),1), aggre(smooth(data16,1),1), aggre(smooth(data17,1),1), aggre(smooth(data18,1),1), aggre(smooth(data19,1),1)),
        (aggre(time_list,4), aggre(smooth(data20,1),1), aggre(smooth(data21,1),1), aggre(smooth(data22,1),1), aggre(smooth(data23,1),1), aggre(smooth(data24,1),1), aggre(smooth(data25,1),1), aggre(smooth(data26,1),1), aggre(smooth(data27,1),1), aggre(smooth(data28,1),1), aggre(smooth(data29,1),1)),
        (aggre(time_list,4), aggre(smooth(data30,1),1), aggre(smooth(data31,1),1), aggre(smooth(data32,1),1), aggre(smooth(data33,1),1), aggre(smooth(data34,1),1), aggre(smooth(data35,1),1), aggre(smooth(data36,1),1), aggre(smooth(data37,1),1), aggre(smooth(data38,1),1), aggre(smooth(data39,1),1)))
self_plot_shadow(data,
          os.path.join(save_path, 'ac-sto_sta.png'),
          xlabel=xlabel,
          ylabel=ylabel,
          legend=legend_without,
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          ylim=ylim,
          yline=3900,
          display=False
          )

# ========== stochastic policy and state value with baseline ==========
data10 = np.load('./Results_dir/sto_sta_baseline/2021-10-20-23-47-20/tar.npy')
data11 = np.load('./Results_dir/sto_sta_baseline/2021-10-20-23-47-29/tar.npy')
data12 = np.load('./Results_dir/sto_sta_baseline/2021-10-20-23-47-41/tar.npy')
data13 = np.load('./Results_dir/sto_sta_baseline/2021-10-20-23-47-46/tar.npy')
data14 = np.load('./Results_dir/sto_sta_baseline/2021-10-21-20-08-16/tar.npy')
data15 = np.load('./Results_dir/sto_sta_baseline/2021-10-21-20-08-20/tar.npy')
data16 = np.load('./Results_dir/sto_sta_baseline/2021-10-21-20-19-22/tar.npy')
data17 = np.load('./Results_dir/sto_sta_baseline/2021-10-26-23-27-39/tar.npy')
data18 = np.load('./Results_dir/sto_sta_baseline/2021-10-26-23-28-04/tar.npy')
data19 = np.load('./Results_dir/sto_sta_baseline/2021-10-26-23-28-45/tar.npy')

data20 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-41/tar.npy')
data21 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-44/tar.npy')
data22 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-47/tar.npy')
data23 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-50/tar.npy')
data24 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-53/tar.npy')
data25 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-44-56/tar.npy')
data26 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-45-02/tar.npy')
data27 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-45-05/tar.npy')
data28 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-45-08/tar.npy')
data29 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-18-45-14/tar.npy')

data30 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-10-28/tar.npy')
data31 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-09/tar.npy')
data32 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-15/tar.npy')
data33 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-33/tar.npy')
data34 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-46/tar.npy')
data35 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-58/tar.npy')
data36 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-17-06/tar.npy')
data37 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-17-19/tar.npy')
data38 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-17-26/tar.npy')
data39 = np.load('./Results_dir/sto_sta_baseline/2021-11-01-15-16-38/tar.npy')

time_list = np.arange(0, MAX_ITERATION, ITER_STEP)
data = ((aggre(time_list,4), aggre(smooth(data10,1),1), aggre(smooth(data11,1),1), aggre(smooth(data12,1),1), aggre(smooth(data13,1),1), aggre(smooth(data14,1),1), aggre(smooth(data15,1),1), aggre(smooth(data16,1),1), aggre(smooth(data17,1),1), aggre(smooth(data18,1),1), aggre(smooth(data19,1),1)),
        (aggre(time_list,4), aggre(smooth(data20,1),1), aggre(smooth(data21,1),1), aggre(smooth(data22,1),1), aggre(smooth(data23,1),1), aggre(smooth(data24,1),1), aggre(smooth(data25,1),1), aggre(smooth(data26,1),1), aggre(smooth(data27,1),1), aggre(smooth(data28,1),1), aggre(smooth(data29,1),1)),
        (aggre(time_list,4), aggre(smooth(data30,1),1), aggre(smooth(data31,1),1), aggre(smooth(data32,1),1), aggre(smooth(data33,1),1), aggre(smooth(data34,1),1), aggre(smooth(data35,1),1), aggre(smooth(data36,1),1), aggre(smooth(data37,1),1), aggre(smooth(data38,1),1), aggre(smooth(data39,1),1)))
self_plot_shadow(data,
          os.path.join(save_path, 'ac-sto_sta_baseline.png'),
          xlabel=xlabel,
          ylabel=ylabel,
          legend=legend_with,
          legend_loc='lower right',
          color_dark=color_dark,
          color_light=color_light,
          ylim=ylim,
          yline=3900,
          display=False
          )
