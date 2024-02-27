"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             Compare DP with different discount factor
             Compare DP with different PEV steps
"""

# =================== load package ====================
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

from lib.plot_utils import EVN_DP_step, EVN_DP_gamma, EVN_DP_RMSE
from lib.utils import makedir
from main_dp_autocar import Alg_DP
from lib.RLParams import RLParams
from matplotlib import pyplot as plt

# <-- User Editable -->
dataSelection = 1           # 1-Use old data  2-Generate new data

# =================== General config ====================
Render = False                    # Show training results
PEV_step_comp = [1, 2, 4]        # different PEV_steps
gamma_comp = [0.85, 0.9, 0.95]   # different discounting factor gamma
maxIteration = 31                # max iteration loop
simulation_num = 10               # run times

# <-- Not Recommended to Modify -->
# ================ Create file folders ==================
ABS_DIR = sys.path[0]
data_save_dir = 'plot_data/'
figure_save_dir = 'figures/'
makedir(ABS_DIR, data_save_dir)
makedir(ABS_DIR, data_save_dir+'data_dp')
datapath = ABS_DIR + '/' + data_save_dir+'data_dp/'
evnpath = '/'.join((ABS_DIR, figure_save_dir+'figures_evaluation'))
True_value_path = ABS_DIR +'/lib/'

def Save(str, a):
    np.save(datapath + str, a)

def Load(str):
    return np.load(datapath + str)

# =================== Start plot ====================
# Data generates
if dataSelection == 2:
    print('Generate New Data.')
    # step compara
    flag = 1
    gamma = 0.90
    R_means_step = np.zeros((len(PEV_step_comp), maxIteration))
    R_means_step_std = np.zeros_like(R_means_step)
    RMSE_step = np.zeros((len(PEV_step_comp), maxIteration))
    RMSE_step_std = np.zeros_like(RMSE_step)

    R_means_gamma = np.zeros((len(gamma_comp), maxIteration))
    R_means_gamma_std = np.zeros_like(R_means_gamma)
    RMSE_gamma = np.zeros((len(gamma_comp), maxIteration))
    RMSE_gamma_std = np.zeros_like(RMSE_gamma)

    V_true = loadmat(True_value_path + 'True_values_DP.mat')
    V_true = V_true['V']
    for step_index in range(len(PEV_step_comp)):
        R_temp = np.zeros((simulation_num, maxIteration))
        RMSE_temp = np.zeros((simulation_num, maxIteration))
        step = PEV_step_comp[step_index]
        for num_run in tqdm(range(simulation_num), desc='gamma=%.2f step=%d'%(gamma, step)):
            params = RLParams('DP', gamma=gamma, maxIteration=maxIteration, PEVsteps=step)
            R_means, RMSE = Alg_DP(flag, params, True, 2, V_true)
            R_temp[num_run] = R_means
            RMSE_temp[num_run] = RMSE
        R_means_step[step_index] = np.average(R_temp, 0)
        RMSE_step[step_index] = np.average(RMSE_temp, 0)
        R_means_step_std[step_index] = np.std(R_temp, 0)
        RMSE_step_std[step_index] = np.std(RMSE_temp, 0)
    Save('R_means_step', R_means_step)
    Save('RMSE_step', RMSE_step)
    Save('R_means_step_std', R_means_step_std)
    Save('RMSE_step_std', RMSE_step_std)
    # gamma compare
    step = 1
    for gamma_index in range(len(gamma_comp)):
        R_temp = np.zeros((simulation_num, maxIteration))
        RMSE_temp = np.zeros((simulation_num, maxIteration))
        gamma = gamma_comp[gamma_index]
        for num_run in tqdm(range(simulation_num), desc='gamma=%.2f step=%d'%(gamma, step)):
            params = RLParams('DP', gamma=gamma, maxIteration=maxIteration, PEVsteps=step)
            R_means, RMSE = Alg_DP(flag, params, True, 2, V_true)
            R_temp[num_run] = R_means
            RMSE_temp[num_run] = RMSE
        R_means_gamma[gamma_index] = np.average(R_temp, 0)
        RMSE_gamma[gamma_index] = np.average(RMSE_temp, 0)
        R_means_gamma_std[gamma_index] = np.std(R_temp, 0)
        RMSE_gamma_std[gamma_index] = np.std(RMSE_temp, 0)
    Save('R_means_gamma', R_means_gamma)
    Save('RMSE_gamma', RMSE_gamma)
    Save('R_means_gamma_std', R_means_gamma_std)
    Save('RMSE_gamma_std', RMSE_gamma_std)
else:
    print('Use Old Data.')
# Load data
DP_reward_n = Load('R_means_step.npy')
DP_reward_gamma = Load('R_means_gamma.npy')
DP_RMSE_n = Load('RMSE_step.npy')
DP_RMSE_gamma = Load('RMSE_gamma.npy')
DP_reward_n_std = Load('R_means_step_std.npy')
DP_reward_gamma_std = Load('R_means_gamma_std.npy')
DP_RMSE_n_std = Load('RMSE_step_std.npy')
DP_RMSE_gamma_std = Load('RMSE_gamma_std.npy')

# print(np.shape(DP_reward_n))
iteration = np.arange(maxIteration)
EVN_DP_step(iteration, PEV_step_comp, DP_reward_n, DP_reward_n_std, evnpath)
EVN_DP_gamma(iteration, gamma_comp, DP_reward_gamma, DP_reward_gamma_std, evnpath)
EVN_DP_RMSE('DP_step_size_RMSE', iteration, PEV_step_comp, DP_RMSE_n, DP_RMSE_n_std, evnpath)
EVN_DP_RMSE('DP_gamma_RMSE', iteration, gamma_comp, DP_RMSE_gamma, DP_RMSE_gamma_std, evnpath)
if Render:
    plt.show()
print('End')
