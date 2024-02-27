"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             Compare Q Learning with different learning rate (alpha)
"""

# =================== load package ====================
import sys
import numpy as np
from scipy.io import loadmat

from lib.plot_utils import EVN_QL_Alpha
from lib.utils import makedir
from main_qlearn_autocar import Alg_QL
from lib.RLParams import RLParams
from matplotlib import pyplot as plt
from lib.env import s_a

# <-- User Editable -->
dataSelection = 1                # 1-Use old data  2-Generate new data

# =================== General config ====================
Render = False                    # Show training results
alpha_comp = [0.05, 0.1, 0.2]    # different learning rate
simulation_num = 10               # Run times for each parameter set

# =================== RL parameters =====================
params = RLParams(name='Qlearn', 
                  alpha=0.1, 
                  epsilon=0.1, 
                  gamma=0.90,
                  maxEpisode=80000, 
                  logPeriod=2000)

# <-- Not Recommended to Modify -->
# ================ Create file folders ==================
ABS_DIR = sys.path[0]
data_save_dir = 'plot_data/'
figure_save_dir = 'figures/'
makedir(ABS_DIR, data_save_dir)
makedir(ABS_DIR, data_save_dir+'data_qlearning')
makedir(ABS_DIR, figure_save_dir)
makedir(ABS_DIR, figure_save_dir + 'figures_evaluation')
datapath = ABS_DIR + '/' + data_save_dir+'data_qlearning/'
evnpath = '/'.join((ABS_DIR, figure_save_dir+'figures_evaluation'))
True_value_path = ABS_DIR +'/lib/'


def Save(str, a):
    np.save(datapath + str, a)


def Load(str):
    return np.load(datapath + str)


# =================== Start plot ====================
# Data generates
# maxIteration = int(params.maxEpisode / params.logPeriod) + 1
if dataSelection == 2:
    print('Generate New Data.')
    Q_true = loadmat(True_value_path + 'True_values_TD.mat')
    Q_true = Q_true['Q']
    ql_R_means = list()
    ql_RMSE = list()
    for alpha in alpha_comp:
        params.alpha = alpha
        R_means = list()
        RMSE = list()
        for run in range(simulation_num):
            Q = np.zeros(s_a)
            _, _, means, rms = Alg_QL(params=params, 
                                      Q=Q,
                                      SimulationN=2,
                                      Q_true=Q_true,
                                      desc='QLearn-alpha %.2f run=%d/%d'%(alpha, run+1, simulation_num))
            R_means.append(means)
            RMSE.append(rms)
        ql_R_means.append(R_means)
        ql_RMSE.append(RMSE)

    ql_R_means = np.array(ql_R_means)
    ql_RMSE = np.array(ql_RMSE)
    Save('R_means', ql_R_means)
    Save('RMSE', ql_RMSE)
else:
    print('Use Old Data.')
ql_R_means = Load('R_means.npy')
ql_RMSE = Load('RMSE.npy')
X = np.arange(ql_R_means.shape[2]) * params.logPeriod
R_means_avg = np.average(ql_R_means, 1)
R_means_std = np.std(ql_R_means, 1)
RMSE_avg = np.average(ql_RMSE, 1)
RMSE_std = np.std(ql_RMSE, 1)


EVN_QL_Alpha(X, R_means_avg, R_means_std, alpha_comp, evnpath, 'QL_Rewards')
EVN_QL_Alpha(X, RMSE_avg, RMSE_std, alpha_comp, evnpath, 'QL_RMSE')
if Render:
    plt.show()
print('End')
