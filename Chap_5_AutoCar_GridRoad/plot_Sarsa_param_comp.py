"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             Compare Sarsa with different PEV-steps
"""

# =================== load package ====================
import sys
import numpy as np
from scipy.io import loadmat

from lib.plot_utils import EVN_Sarsa_step
from lib.utils import makedir
from main_sarsa_autocar import Alg_Sarsa
from lib.RLParams import RLParams
from matplotlib import pyplot as plt
from lib.env import s_a

# <-- User Editable -->
dataSelection = 1           # 1-Use old data  2-Generate new data

# =================== General config ====================
Render = False               # Show training results
PEV_step_comp = [1, 8, 32]  # different PEV_steps
simulation_num = 10          # Run times for each parameter set

# =================== RL parameters =====================
params = RLParams(name='Sarsa', 
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
makedir(ABS_DIR, data_save_dir+'data_sarsa')
makedir(ABS_DIR, figure_save_dir)
makedir(ABS_DIR, figure_save_dir + 'figures_evaluation')
datapath = ABS_DIR + '/' + data_save_dir+'data_sarsa/'
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
    sarsa_R_means = list()
    sarsa_RMSE = list()
    for pevstep in PEV_step_comp:
        params.PEVsteps = pevstep
        R_means = list()
        RMSE = list()
        for run in range(simulation_num):
            Q = np.zeros(s_a)
            _, _, means, rms = Alg_Sarsa(params=params, 
                                         Q=Q,
                                         SimulationN=2,
                                         Q_true=Q_true,
                                         desc='Sarsa-%d pair run=%d/%d'%(pevstep, run+1, simulation_num))
            R_means.append(means)
            RMSE.append(rms)
        sarsa_R_means.append(R_means)
        sarsa_RMSE.append(RMSE)

    sarsa_R_means = np.array(sarsa_R_means)
    sarsa_RMSE = np.array(sarsa_RMSE)
    Save('R_means', sarsa_R_means)
    Save('RMSE', sarsa_RMSE)
else:
    print('Use Old Data.')
sarsa_R_means = Load('R_means.npy')
sarsa_RMSE = Load('RMSE.npy')
X = np.arange(sarsa_R_means.shape[2]) * params.logPeriod
R_means_avg = np.average(sarsa_R_means, 1)
R_means_std = np.std(sarsa_R_means, 1)
RMSE_avg = np.average(sarsa_RMSE, 1)
RMSE_std = np.std(sarsa_RMSE, 1)

EVN_Sarsa_step(X, R_means_avg, R_means_std, PEV_step_comp, evnpath, 'Sarsa_Rewards')
EVN_Sarsa_step(X, RMSE_avg, RMSE_std, PEV_step_comp, evnpath, 'Sarsa_RMSE')
if Render:
    plt.show()
print('End')
