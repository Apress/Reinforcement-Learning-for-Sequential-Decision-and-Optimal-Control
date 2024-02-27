"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  RL Example for Autonomous Driving in Curved Grid Road
             % [Method] Dynamic Programming (DP)
             % DP with single-PEV (Standard Value Iteration)
                (1) DP Method with discount rate = gamma
                (2) Initial Value = Terminial Value of last RL iteration
                (3) Ending Condition: Policy does NOT change or RL iteration > maxIteration
                (4) Policy Evaluation contains one step for every iteration
             % DP with multi-PEV + fixed step
                (1) DP Method with discount rate = gamma
                (2) Initial Value = Terminial Value of last RL iteration
                (3) Ending Condition: Policy does NOT change or RL iteration > maxIteration
                (4) Policy Evaluation contains several steps for every iteration
             % DP with multi-PEV + fixed error
                (1) DP Method with discount rate = gamma
                (2) Initial Value = Terminial Value of last RL iteration
                (3) Ending Condition: Policy does NOT change or RL iteration > maxIteration
                (4) Policy Evaluation will stop when the norm of delta_V < 0.01 for every iteration
"""

# =================== load package ====================
import os
import sys
from time import localtime
import numpy as np
from scipy.io import loadmat

from lib.plot_utils import Draw_Policy, Draw_Route, Draw_Value, EVN_DP_step, EVN_DP_RMSE
from lib.utils import precondition, nextstate, reward, isoutside, headlocation, makedir
from lib.env import roadColumn, roadRow, carDirection, state_size, action_size, laneBoundary, gridNum
from lib.RLParams import RLParams
from matplotlib import pyplot as plt

# <-- User Editable -->
# =================== General config ====================
Selection = 2  # Choose a DP method
               # 1 - DP with single-PEV
               # 2 - DP with multi-PEV + fixed step
               # 3 - DP with multi-PEV + fixed error
               # See description for details

Render = False # Show training results

# =================== RL parameters =====================
params = RLParams(name='DP', gamma=0.90, maxIteration=100, PEVsteps=3)


# <-- Not Recommended to Modify -->
# ================ Create file folders ==================
ABS_DIR = sys.path[0]
fig_dir = 'figures/'
True_value_path = ABS_DIR +'/lib/'
makedir(ABS_DIR, fig_dir)
save_dir = fig_dir+'figures_dp/'
makedir(ABS_DIR, save_dir)
data_dir = ABS_DIR + '/results/'

# =================== Initialize environment ====================
deadstate_flag, deadaction_flag = precondition(state_size, action_size, laneBoundary)  # Explore feasible region

# =================== Define record function ====================
def record_DP(SimulationN, policy, V, V_true):
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
                action = policy[tuple(s_now)]
                s_next = nextstate(s_now, action)
                R_sum += reward(s_now, action, s_next)
                s_h_next = headlocation(s_next)
                if s_h_next[0] == roadColumn - 1 or s_next[0] == roadColumn - 1:
                    break
                s_now = s_next.copy()
            R_num += 1
            R_sums = R_sums * R_num / (R_num + 1) + R_sum / (R_num + 1)

    R_means = R_sums
    V_error = (V_true - V) ** 2
    RMSE = (sum(V_error.flatten()) / carDirection / gridNum) ** 0.5
    return R_means, RMSE


# =================== Define DP algorithm ====================
def Alg_DP(flag, params:RLParams, record=False, SimulationN=0, V_true=None):

    step = params.PEVsteps
    if flag == 2:
        step = 1
    gamma = params.gamma
    maxIteration = params.maxIteration
    R_means = np.zeros(maxIteration)
    RMSE = np.zeros(maxIteration)

    # initialize value function and policy
    V = np.zeros(state_size)
    V_old = np.zeros(state_size)
    policy = np.ones(state_size, dtype=int) * -1
    policy_old = np.ones(state_size, dtype=int) * -1

    for idx in np.ndindex(roadColumn - 1, roadRow, carDirection):
        if deadstate_flag[idx] == 1:
            policy[idx] = -1
            continue
        else:
            policy[idx] = np.random.randint(3)
            while deadaction_flag[idx][policy[idx]] == 1:
                policy[idx] = np.random.randint(3)

    index_stop = 1

    # ------------------------------------
    # Dynamic Programming (DP) Iteration
    # ------------------------------------
    Q = np.zeros(3)
    q = [0, 0, 0]
    q_flag = [-10., -10., -10.]

    for iteration in range(maxIteration):

        if not record:
            print('RL interation = %i' % (iteration + 1))
        # Policy Evaluation(PEV)
        first = 1
        index_vstop = 0
        means, MSE = record_DP(SimulationN, policy, V, V_true)
        R_means[iteration] = means
        RMSE[iteration] = MSE
        while (index_vstop == 0 and flag == 0) or first == 1:
            first = 0
            for _ in range(step):
                V_old = V.copy()
                for idx in np.ndindex(roadColumn - 1, roadRow, carDirection):
                    s = np.array(idx)
                    if deadstate_flag[idx] == 0:
                        s_next = nextstate(s, policy[idx])
                        R = reward(s, policy[idx], s_next)
                        V[idx] = R + gamma * V[tuple(s_next)]

                for s2 in range(laneBoundary[1, -2], laneBoundary[0, -2] + 1):
                    for s3 in range(carDirection):
                        s_terminal = np.array([roadColumn - 2, s2, s3])
                        error_now = isoutside(s_terminal, laneBoundary)
                        if error_now == 1:
                            s_terminal_h = headlocation(s_terminal)
                            if s_terminal_h[0] == roadColumn - 1:
                                V[roadColumn - 2, s2, s3] = 0

                # error of value function
                delta_V = V - V_old
                delta_Vplus = np.linalg.norm(delta_V.flatten()) + 1
                if delta_Vplus >= 1 and delta_Vplus < 1.01:
                    index_vstop = 1
                else:
                    index_vstop = 0

        # Policy Improvement (PIM)
        policy_old = policy.copy()
        for idx in np.ndindex(roadColumn - 1, roadRow, carDirection):
            s = np.array(idx)
            if deadstate_flag[idx] == 0:
                n = -1
                for action in range(3):
                    if deadaction_flag[idx][action] == 0:
                        s_next = nextstate(s, action)
                        R = reward(s, action, s_next)
                        Q[action] = R + gamma * V[tuple(s_next)]
                        n = n + 1
                        q[n] = action
                        q_flag[n] = Q[action]
                number2 = q_flag.index(max(q_flag[0:n + 1]))
                policy[idx] = q[number2]

        delta_policy = policy - policy_old
        index_stop = np.linalg.norm(delta_policy.flatten())
        if index_stop == 0:
            if record:
                R_means[iteration+1:] = means
                RMSE[iteration+1:] = MSE
            break

    if record:
        return R_means, RMSE
    else:
        for s2 in range(laneBoundary[1, -2], laneBoundary[0, -2] + 1):
            for s3 in range(carDirection):
                s_terminal = np.array([roadColumn - 2, s2, s3])
                error_now = isoutside(s_terminal, laneBoundary)
                if error_now == 1:
                    s_terminal_h = headlocation(s_terminal)
                    V[roadColumn - 2, s2, s3] = 0
                    policy[roadColumn - 2, s2, s3] = -1
        return policy, V, R_means[0:iteration+1], RMSE[0:iteration+1], iteration+1


# =================== Run DP algorithm ====================
if __name__ == '__main__':
    def Case1():  # Case1: DP with single-PEV
        os.system('cls')
        print('-------- DP with single-PEV (i.e., value iteration)')
        folderName = save_dir + 'DP_singlePEV'
        makedir(ABS_DIR, folderName)
        return 2, folderName


    def Case2():
        os.system('cls')
        print('-------- DP with multi-PEV + fixed step (i.e. policy iteration)')
        folderName = save_dir + 'DP_multiPEV_fixStep'
        makedir(ABS_DIR, folderName)
        return 1, folderName


    def Case3():
        os.system('cls')
        print('-------- DP with multi-PEV + minimum error (i.e., policy iteration)')
        folderName = save_dir + 'DP_multiPEV_minErr'
        makedir(ABS_DIR, folderName)
        return 0, folderName


    Dic_Selection = {
        1: Case1,
        2: Case2,
        3: Case3
    }

    flag, folder = Dic_Selection[Selection]()
    V_true = loadmat(True_value_path + 'True_values_DP.mat')
    V_true = V_true['V']
    policy, V, R_means, RMSE, num_iter = Alg_DP(flag, params=params, SimulationN=2, V_true=V_true)
    print('End of RL')
    # -------------------------------------------
    # Draw optimal policy
    # -------------------------------------------
    print('Draw_Policy...')
    Draw_Policy(policy, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw optimal value function
    # -------------------------------------------
    Using_V_value = 1
    print('Draw_Value...')
    Draw_Value(V, policy, Using_V_value, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw a specific route with initial states
    # -------------------------------------------
    print('Draw_Route...')
    Draw_Route(policy, ('/'.join([ABS_DIR, folder])))
    # -------------------------------------------
    # Draw training results
    # -------------------------------------------
    print('Draw_Results...')
    iteration = np.arange(num_iter)
    EVN_DP_step(iteration, [params.PEVsteps], [R_means], np.zeros_like(R_means), ('/'.join([ABS_DIR, folder])), False)
    EVN_DP_RMSE('DP_RMSE',iteration, [params.PEVsteps], [RMSE], np.zeros_like(R_means), ('/'.join([ABS_DIR, folder])), False)
    mytime = localtime()
    save_name = '%s_%s%s%s_%s%s'%(params.name, mytime.tm_year, mytime.tm_mon, mytime.tm_mday, mytime.tm_hour, mytime.tm_min)
    np.savez(data_dir+save_name, R_means=R_means, RMSE=RMSE)
    if Render:
        plt.show()
    # print(R_means)
    print('End')
