"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             % [Method] Temporal-Difference (TD) RL
             % SARSA (On-Policy TD)
                 (1) TD with discount rate = gamma & learning rate = alpha
                 (2) e-Greedy Policy with fixed epsilon
                 (3) Ending Condition: episode > maxEpisode
                 (4) Policy will be evaluated for each state pair with the initial Q-value from the last state pair
"""

# =================== load package ====================
import os
import sys
import numpy as np
from scipy.io import loadmat
from tqdm.std import tqdm
from time import localtime

from lib.plot_utils import Draw_Policy, Draw_Route, Draw_Value, EVN_TD
from lib.utils import makedir, precondition, headlocation, get_action, nextstate, reward
from lib.env import roadColumn, roadRow, carDirection, state_size, action_size, laneBoundary, gridNum, s_a
from lib.RLParams import RLParams
from matplotlib import pyplot as plt

# <-- User Editable -->
# =================== General config ====================
Render = True    # Show training results

# =================== RL parameters =====================
params_sarsa = RLParams(name='Sarsa', 
                        alpha=0.1, 
                        epsilon=0.1, 
                        gamma=0.98, 
                        maxEpisode=100000, 
                        PEVsteps=1, 
                        logPeriod=2000)


# <-- Not Recommended to Modify -->
# ================ Create file folders ==================
ABS_DIR = sys.path[0]
fig_dir = 'figures/'
makedir(ABS_DIR, fig_dir)
save_dir = fig_dir+'figures_sarsa/'
makedir(ABS_DIR, save_dir)
data_dir = ABS_DIR + '/results/'
True_value_path = ABS_DIR +'/lib/'

# =================== Initialize environment ====================
deadstate_flag, deadaction_flag = precondition(state_size, action_size, laneBoundary)


# =================== Define record function ====================
def record_Sarsa(epsilon, SimulationN, Q, Q_true):
    R_num = 0
    R_sums = 0
    for _ in range(SimulationN):
        for idx in np.ndindex(roadColumn - 1, roadRow, carDirection):
            s_now = np.array(idx)
            s_h_now = headlocation(s_now)
            if s_h_now[0] == roadColumn - 1 or deadstate_flag[tuple(s_now)] == 1:
                continue
            # s_start = s_now.copy()
            R_sum = 0
            while True:
                action = get_action(Q, s_now, epsilon, deadaction_flag)
                s_next = nextstate(s_now, action)
                R_sum += reward(s_now, action, s_next)
                s_h_next = headlocation(s_next)
                if s_h_next[0] == roadColumn - 1 or s_next[0] == roadColumn - 1:
                    break
                s_now = s_next.copy()

            R_sums = R_sums * R_num / (R_num + 1) + R_sum / (R_num + 1)
            R_num += 1

    R_means = R_sums
    Q_error = (Q_true - Q) ** 2
    RMSE = (sum(Q_error.flatten()) / carDirection / gridNum) ** 0.5
    return R_means, RMSE

# =================== Define Sarsa algorithm ====================
def Alg_Sarsa(params:RLParams, Q, SimulationN=0, Q_true=None, desc:str=None):
    Q_old = Q.copy()
    alpha = params.alpha
    maxEpisode = params.maxEpisode
    gamma = params.gamma
    epsilon = params.epsilon
    logPeriod = params.logPeriod

    index_available = np.zeros(state_size, dtype=int)

    # ------------------------------------------
    # TD Iteration
    # ------------------------------------------
    td_iter = range(maxEpisode + 1)
    if desc is None:
        desc = params.name
    td_iter = tqdm(td_iter, desc='%s'%desc)
    policy = np.ones(state_size, dtype=int) * -1
    means_list = list()
    RMSE_list = list()
    step_count = 0
    for td_i in td_iter:
        # Random intial state
        if  td_i % logPeriod == 0:
            mes, rms = record_Sarsa(epsilon, SimulationN, Q, Q_true)
            means_list.append(mes)
            RMSE_list.append(rms)
        s_now = np.array(
            [np.random.randint(roadColumn - 1), np.random.randint(roadRow), np.random.randint(carDirection)])
        s_h_now = headlocation(s_now)
        # Reselect when the car is outside of the lane or the head is at the terminal
        while (s_h_now[0] == roadColumn - 1 or deadstate_flag[s_now[0], s_now[1], s_now[2]] == 1):
            s_now = np.array(
                [np.random.randint(roadColumn - 1), np.random.randint(roadRow), np.random.randint(carDirection)])
            s_h_now = headlocation(s_now)
        s_start = s_now.copy()
        action_now = get_action(Q, s_now, epsilon, deadaction_flag)
        # Sampling
        # counter = 0
        while True:
            s_next = nextstate(s_now, action_now)
            R = reward(s_now, action_now, s_next)
            # Gain next action, 1:left; 2:right; 3:keep
            if step_count == params.PEVsteps:
                step_count = 0
                Q_old = Q.copy()
            action_next = get_action(Q_old, s_next, epsilon, deadaction_flag)
            # Update Q-value
            delta = R + gamma * Q[s_next[0], s_next[1], s_next[2], action_next] - Q[
                s_now[0], s_now[1], s_now[2], action_now]
            Q[s_now[0], s_now[1], s_now[2], action_now] = Q[s_now[0], s_now[1], s_now[2], action_now] + alpha * delta
            step_count += 1
            # if the car is at the terminal then break
            s_h_next = headlocation(s_next)
            if s_h_next[0] == roadColumn - 1 or s_next[0] == roadColumn - 1:
                index_available[s_start[0], s_start[1], s_start[
                    2]] = 1  # Signal that the car can reach the terminal from the start state
                break
            s_now = s_next
            action_now = action_next

    # Gain the final policy by greedy policy
    for idx in np.ndindex(roadColumn - 1, roadRow, carDirection):
        s = np.array(idx)
        if deadstate_flag[idx] == 0:
            policy[idx] = get_action(Q, s, 0, deadaction_flag)

    policy[index_available == 0] = -1
    return policy, Q, np.array(means_list), np.array(RMSE_list)

# =================== Run Sarsa algorithm ====================
if __name__ == '__main__':
    os.system('cls')

    folder = save_dir
    makedir(ABS_DIR, folder)
    Q = np.zeros(s_a)
    Q_true = loadmat(True_value_path + 'True_values_TD.mat')
    Q_true = Q_true['Q']
    policy, Q, R_means, RMSE = Alg_Sarsa(params_sarsa, Q, SimulationN=2, Q_true=Q_true)
    mytime = localtime()
    save_name = '%s_%s%s%s_%s%s'%(params_sarsa.name, mytime.tm_year, mytime.tm_mon, mytime.tm_mday, mytime.tm_hour, mytime.tm_min)
    np.savez(data_dir+save_name, R_means=R_means, RMSE=RMSE)
    X = np.arange(len(R_means)) * params_sarsa.logPeriod
    print('End of RL')
    # -------------------------------------------
    # Draw optimal policy
    # -------------------------------------------
    print('Draw_Policy...')
    Draw_Policy(policy, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw optimal value function
    # -------------------------------------------
    print('Draw_Value...')
    Draw_Value(Q, policy, 0, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw a specific route with initial states
    # -------------------------------------------
    print('Draw_Route...')
    Draw_Route(policy, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw trainning results
    # -------------------------------------------
    print('Draw_Results...')
    EVN_TD(X, QL=R_means, Sarsa=None, savepath=('/'.join([ABS_DIR, folder])), name='Rewards', mode=False)
    EVN_TD(X, QL=RMSE, Sarsa=None, savepath=('/'.join([ABS_DIR, folder])), name='RMS Error', mode=False)
    if Render:
        plt.show()
    print('End')