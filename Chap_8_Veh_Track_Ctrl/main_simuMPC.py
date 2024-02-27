"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 8: Simulate MPC controller and ADP controller for comparison

"""


import math
import os
import time
from datetime import datetime
from matplotlib import markers

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from config import MPCConfig
from dynamics import TrackingEnv
from network import Actor, Critic
from solver import Solver
from utils import idplot, cm2inch, calRelError

def simulationReal(MPCStep, ADP_dir, simu_dir, seed=0, init_bias=True):
    plotDelete = 0
    figure_dir = simu_dir + '/Figures_bias' if init_bias else simu_dir + '/Figures'
    os.makedirs(figure_dir, exist_ok=True)
    data_dir = simu_dir + '/Data'
    os.makedirs(data_dir, exist_ok=True)
    env = TrackingEnv()
    env.seed(seed)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    if init_bias:
        initialState = env.resetGiven(1, MPCtest=True)
    else:
        initialState = env.resetSpecificCurve(1, curveType = 'sine', noise = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    timeADP = 0
    while(count < env.testStepReal):
        stateADPList = np.append(stateADPList, stateAdp.numpy()) # [v, omega, x, y, phi, xr, yr, phir]
        relState = env.relStateCal(stateAdp)
        timeStart = time.time()
        controlAdp = policy(relState).detach()
        timeADP += time.time() - timeStart
        stateAdp, reward, done = env.stepReal(stateAdp, controlAdp, curveType = 'sine')
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1

    print('METHODS: ADP, CALCULATION TIME: {:.3f}'.format(timeADP) + 's')
    stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
    controlADPList = np.reshape(controlADPList, (-1, actionDim))

    stateADPList = np.delete(stateADPList, range(plotDelete), 0)
    controlADPList = np.delete(controlADPList, range(plotDelete), 0)
    rewardADP = np.delete(rewardADP, range(plotDelete), 0)

    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(data_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")


    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    timeMPCAll = []
    timeMPCAll.append(timeADP)
    for mpcstep in MPCStep:
        print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].clone().tolist()
        stateMpc = tempstate[0:5] # v,omega,x,y,phi
        refStateMpc = tempstate[5:8] # xr,yr,phir
        count = 0
        controlMPCList = np.empty(0)
        stateMPCList = np.empty(0)
        rewardMPC = np.empty(0)
        timeMPC = 0
        while(count < env.testStepReal):
            # MPC
            timeStart = time.time()
            _, control = solver.MPCSolver(stateMpc, refStateMpc, mpcstep, isReal = False)
            timeMPC += time.time() - timeStart
            stateMPCList = np.append(stateMPCList, np.array(stateMpc))
            stateMPCList = np.append(stateMPCList, np.array(refStateMpc))
            action = control[0].tolist()
            reward = env.calReward(stateMpc + refStateMpc,action,MPCflag=1)
            temp = env.vehicleDynamic(
                stateMpc[2], stateMpc[3], stateMpc[4], env.refV, stateMpc[0], stateMpc[1], 0, action[0], MPCflag=1)
            stateMpc[2:5] = temp[:3] # x, y, phi
            stateMpc[:2] = temp[4:6] # v, omega
            refStateMpc = env.refDynamicReal(refStateMpc, MPCflag=1, curveType = 'sine')
            rewardMPC = np.append(rewardMPC, reward)
            controlMPCList = np.append(controlMPCList, control[0])
            count += 1
        print('METHODS: MPC {} steps, CALCULATION TIME: {:.3f}'.format(mpcstep, timeMPC) + 's')
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        stateMPCList = np.delete(stateMPCList, range(plotDelete), 0)
        controlMPCList = np.delete(controlMPCList, range(plotDelete), 0)
        rewardMPC = np.delete(rewardMPC, range(plotDelete), 0)

        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(data_dir + "/simulationRealMPC_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")

        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)
        timeMPCAll.append(timeMPC)
        # print("Overall Cost for {} Steps, MPC: {:.3f}, ADP: {:.3f}".format(env.testStepReal, rewardMPC, rewardADP.item()))

    if os.path.exists(data_dir + "/time.csv")==False:
        with open(data_dir + "/time.csv", 'ab') as f:
            np.savetxt(f, np.array([timeMPCAll]), delimiter=',', fmt='%.6f', comments='', header="ADP,MPC-"+",MPC-".join([str(m) for m in MPCStep]))
    else:
        with open(data_dir + "/time.csv", 'ab') as f:
            np.savetxt(f, np.array([[0, timeMPC, env.testStepReal]]), delimiter=',', fmt='%.6f', comments='')

    # Cal relative error
    errorSaveList = []
    # Steering Angle
    ADPData = controlADPList[:,0]
    MPCData = controlMPCAll[-1][:, 0]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Steering Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Lateral Position
    # ADPData =  - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    # MPCData = - (stateMPCAll[-1][:,2] - stateMPCAll[-1][:,5]) * np.sin(stateMPCAll[-1][:,7]) + (stateMPCAll[-1][:,3] - stateMPCAll[-1][:,6]) * np.cos(stateMPCAll[-1][:,7])
    ADPData = stateADPList[:, 3]
    MPCData = stateMPCAll[-1][:, 3]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Lateral Position', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Heading Angle
    ADPData = stateADPList[:,4]
    MPCData = stateMPCAll[-1][:, 4]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Heading Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Utility  Function
    ADPData = rewardADP
    MPCData = rewardMPCAll[-1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Utility  Function', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    if os.path.exists(data_dir + "/RelError.csv")==False:
        with open(data_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Steering Angle mean,max,Lateral Position mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(data_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')

    # Plot
    # stateAll = [v, omega, x, y, phi, xr, yr, phir]
    # y v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,3]
    yMPC = [mpc[:,3] for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Lateral position [m]'
    title = '1-Lat Position'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, legend_loc='lower left')

    # lateral error v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = 100 * (- (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7]))
    yMPC = [100 * (- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7])) for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Lateral position error [cm]'
    title = '1-Lat position error'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, isError = True, legend_loc='upper left') # , isError = True

    # phi v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle [°]'
    title = '1-Heading angle'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, legend_loc='lower left')

    # phi error v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle error [°]'
    title = '1-Heading angle error'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, isError = True, legend_loc='lower left') # , isError = True

    # utility v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Utility'
    title = '1-Utility func'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, legend_loc='lower left')

    # delta v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = controlADPList[:,0] * 180/np.pi
    yMPC = [mpc[:,0]* 180/np.pi for mpc in controlMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Steering angle [°]'
    title = '1-Control input'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, legend_loc='upper left')

    OP = controlMPCAll[-1][:,0] * 180/np.pi
    yADP = yADP - OP
    yMPC = [mpc - OP for mpc in yMPC]
    xName = 'Travel dist [m]'
    yName = 'Steering angle error [°]'
    title = '1-Control input error'
    idplot(xADP, [mpc for mpc in xMPC], yADP, yMPC, MPCStep, xName, yName, figure_dir, title, isError=True, legend_loc='upper right')


def simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 0, seed = 0):
    # apply in virtual time
    plotDelete = 0
    figure_dir = simu_dir + '/Figures'
    os.makedirs(simu_dir, exist_ok=True)
    data_dir = simu_dir + '/Data'
    os.makedirs(data_dir, exist_ok=True)
    env = TrackingEnv()
    env.seed(seed)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.resetGiven(1, MPCtest = True) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < env.testStepVirtual):
        stateADPList = np.append(stateADPList, stateAdp.numpy()) # [v,omega,x,y,phi,xr,yr,phir]
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done = env.stepVirtual(stateAdp, controlAdp)
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1
    stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
    controlADPList = np.reshape(controlADPList, (-1, actionDim))
    stateADPList = np.delete(stateADPList, range(plotDelete), 0)
    controlADPList = np.delete(controlADPList, range(plotDelete), 0)
    rewardADP = np.delete(rewardADP, range(plotDelete), 0)
    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(data_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")

    # MPC
    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    for mpcstep in MPCStep:
        print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].clone().tolist()
        stateMpc = tempstate[:5] # [v,omega,x,y,phi,xr,yr,phir]
        refStateMpc = tempstate[5:8]
        count = 0
        controlMPCList = np.empty(0)
        stateMPCList = np.empty(0)
        rewardMPC = np.empty(0)
        _, control = solver.MPCSolver(stateMpc, refStateMpc, mpcstep, isReal=False)
        while(count < mpcstep):
            stateMPCList = np.append(stateMPCList, np.array(stateMpc))
            stateMPCList = np.append(stateMPCList, np.array(refStateMpc))
            action = control[count].tolist()
            reward = env.calReward(stateMpc + refStateMpc,action,MPCflag=1)
            temp = env.vehicleDynamic(
                stateMpc[2], stateMpc[3], stateMpc[4], env.refV, stateMpc[0], stateMpc[1], 0, action[0], MPCflag=1)
            stateMpc[2:5] = temp[:3] # x, y, phi
            stateMpc[:2] = temp[4:6] # v, omega
            refStateMpc = env.refDynamicVirtual(refStateMpc, MPCflag=1)
            rewardMPC = np.append(rewardMPC, reward)
            controlMPCList = np.append(controlMPCList, control[count])
            count += 1
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        stateMPCList = np.delete(stateMPCList, range(plotDelete), 0)
        controlMPCList = np.delete(controlMPCList, range(plotDelete), 0)
        rewardMPC = np.delete(rewardMPC, range(plotDelete), 0)
        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(data_dir + "/simulationVirtualMPC_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)

    # Cal relative error
    errorSaveList = []
    # Steering Angle
    ADPData = controlADPList[:,0]
    MPCData = controlMPCAll[-1][:, 0]
    if len(ADPData) > len(MPCData):
        ADPData = ADPData[:len(MPCData)]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Steering Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Lateral Position
    # ADPData =  - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    # MPCData = - (stateMPCAll[-1][:,2] - stateMPCAll[-1][:,5]) * np.sin(stateMPCAll[-1][:,7]) + (stateMPCAll[-1][:,3] - stateMPCAll[-1][:,6]) * np.cos(stateMPCAll[-1][:,7])
    ADPData = stateADPList[:, 3]
    MPCData = stateMPCAll[-1][:, 3]
    if len(ADPData) > len(MPCData):
        ADPData = ADPData[:len(MPCData)]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Lateral Position', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Heading Angle
    ADPData = stateADPList[:,4] - stateADPList[:,7]
    MPCData = stateMPCAll[-1][:, 4] - stateMPCAll[-1][:, 7]
    if len(ADPData) > len(MPCData):
        ADPData = ADPData[:len(MPCData)]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Heading Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Utility  Function
    ADPData = rewardADP
    MPCData = rewardMPCAll[-1]
    if len(ADPData) > len(MPCData):
        ADPData = ADPData[:len(MPCData)]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Utility  Function', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    if os.path.exists(data_dir + "/RelError.csv")==False:
        with open(data_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Steering Angle mean,max,Lateral Position mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(data_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')


    # Plot
    # stateAll = [v, omega, x, y, phi, xr, yr, phir]
    # delta v.s. t
    xADP = np.arange(0, len(controlADPList[:,0]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,0]) * env.T, env.T) for mpc in controlMPCAll]
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Steering Angle [°]'
    title = '2-Control-0 Step'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, denseMark= True, condition='virtual')

    # phi error v.s. t
    xADP = np.arange(0, len(stateADPList[:,4]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,4]) * env.T, env.T) for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Heading angle error [°]'
    title = '2-Head angle-0 Step'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, denseMark= True, condition='virtual')

    # lateral error v.s. t
    yADP = 100 * (- (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7]))
    yMPC = [100 * (- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7])) for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'Lateral position error [cm]'
    title = '2-Lat position-0 Step'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, denseMark= True, condition='virtual')

    # phi v.s. lateral error
    xADP =  100 * (-(stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7]))
    xMPC = [100 * (- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7])) for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Lateral position error [cm]'
    yName = 'Heading angle error [°]'
    title = '2-Phase plot-0 Step'
    idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, figure_dir, title, denseMark= True, condition='virtual')

def main(log_dir):
    config = MPCConfig()
    MPCStep = config.MPCStep

    simu_dir = log_dir + "-Simu"
    os.makedirs(simu_dir, exist_ok=True)

    # 4. 真实时域ADP、MPC应用
    simulationReal(MPCStep, log_dir, simu_dir, init_bias=False)

    # 5. 虚拟时域ADP、MPC应用
    simulationVirtual(MPCStep, log_dir, simu_dir)

if __name__ == '__main__':
    ADP_dir = 'Results/2022-05-09-20-57-06'
    main(ADP_dir)