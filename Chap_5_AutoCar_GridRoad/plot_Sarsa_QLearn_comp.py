"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             Compare the performance of Sarsa Q-learning and On-policy MC
"""

# =================== load package ====================
from matplotlib import pyplot as plt
from lib.RLParams import RLParams
import numpy as np
import sys
from main_sarsa_autocar import Alg_Sarsa
from main_qlearn_autocar import Alg_QL
from scipy.io import loadmat
from lib.utils import makedir
from lib.plot_utils import EVN_TD
from lib.env import s_a

# <-- User Editable -->
dataSelection = 1     # 1-Use old data  2-Generate new data

# ================== General config ====================
Render = False         # Show training results
simulation_num = 10    # Run times for each Algorithm

# =================== RL parameters ====================
maxEpisode = 80000
logPeriod = 2000
QLparams = RLParams(name='QLearning',
                    alpha=0.1, 
                    epsilon=0.1, 
                    gamma=0.90,
                    maxEpisode=maxEpisode, 
                    logPeriod=logPeriod)

SAparams = RLParams(name='Sarsa', 
                    alpha=0.1, 
                    epsilon=0.1, 
                    gamma=0.90,
                    maxEpisode=maxEpisode,
                    logPeriod=logPeriod)


# <-- Not Recommended to Modify -->
# ================ Create file folders ==================
ABS_DIR = sys.path[0]
data_save_dir = 'plot_data/'
makedir(ABS_DIR, data_save_dir)
figure_save_dir = 'figures/'
makedir(ABS_DIR, figure_save_dir)
makedir(ABS_DIR, figure_save_dir + 'figures_evaluation')
makedir(ABS_DIR, data_save_dir + 'data_td')
makedir(ABS_DIR, data_save_dir + 'data_td/Q_Learning')
makedir(ABS_DIR, data_save_dir + 'data_td/Sarsa')
temp_dir = ABS_DIR + '/' + data_save_dir
TDdatapath = [temp_dir + 'data_td/', temp_dir + 'data_td/Q_Learning/', temp_dir + 'data_td/Sarsa/']
evnpath = '/'.join((ABS_DIR, figure_save_dir+'figures_evaluation'))
True_value_path = ABS_DIR +'/lib/'


def SaveTD(selection, str, a):
    np.save(TDdatapath[selection] + str, a)


def LoadTD(selection, str):
    return np.load(TDdatapath[selection] + str)


# ================ Start Plot ================
maxIteration = int(maxEpisode / logPeriod) + 1
if dataSelection == 2:
    print('Generate New Data.')
    Q_true = loadmat(True_value_path + 'True_values_TD.mat')
    Q_true = Q_true['Q']
    for alg in [1, 2]:
        R_means = np.zeros((simulation_num, maxIteration))
        RMSE = np.zeros((simulation_num, maxIteration))
        for run in range(simulation_num):
            Q = np.zeros(s_a)
            if alg == 1:
                _, _, R_means[run], RMSE[run] = Alg_QL(params=QLparams,
                                                       Q=Q,
                                                       SimulationN=2,
                                                       Q_true=Q_true,
                                                       desc='Qlearn run=%d/%d'%(run+1, simulation_num))
            else:
                _, _, R_means[run], RMSE[run] = Alg_Sarsa(params=SAparams,
                                                          Q=Q,
                                                          SimulationN=2,
                                                          Q_true=Q_true,
                                                          desc='Sarsa run=%d/%d'%(run+1, simulation_num))
        SaveTD(alg, 'R_means', R_means)
        SaveTD(alg, 'RMSE', RMSE)
else:
    print('Use Old Data.')
X = np.arange(maxIteration) * logPeriod
QL_reward = {}
QL_reward['R_means'] = LoadTD(1, 'R_means.npy')
QL_reward['average'] = np.average(QL_reward['R_means'], 0)
QL_reward['std'] = np.std(QL_reward['R_means'], 0)
QL_RMSE = {}
QL_RMSE['RMSE'] = LoadTD(1, 'RMSE.npy')
QL_RMSE['average'] = np.average(QL_RMSE['RMSE'], 0)
QL_RMSE['std'] = np.std(QL_RMSE['RMSE'], 0)

Sarsa_reward = {}
Sarsa_reward['R_means'] = LoadTD(2, 'R_means.npy')
Sarsa_reward['average'] = np.average(Sarsa_reward['R_means'], 0)
Sarsa_reward['std'] = np.std(Sarsa_reward['R_means'], 0)
Sarsa_RMSE = {}
Sarsa_RMSE['RMSE'] = LoadTD(2, 'RMSE.npy')
Sarsa_RMSE['average'] = np.average(Sarsa_RMSE['RMSE'], 0)
Sarsa_RMSE['std'] = np.std(Sarsa_RMSE['RMSE'], 0)


EVN_TD(X, QL_reward['average'], Sarsa_reward['average'], QL_reward['std'], Sarsa_reward['std'], evnpath, 'QL-SARSA_Rewards')
EVN_TD(X, QL_RMSE['average'], Sarsa_RMSE['average'],QL_RMSE['std'], Sarsa_RMSE['std'], evnpath, 'QL-SARSA_RMSE')
if Render:
    plt.show()
print('Plot finished!')
