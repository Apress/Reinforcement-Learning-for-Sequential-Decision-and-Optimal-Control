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
from myenv import TrackingEnv
from network import Actor, Critic
from solver import Solver

def simulationReal(MPCStep, ADP_dir, simu_dir,seed=0):
    plotDelete = 0
    # apply in real time
    env = TrackingEnv()
    env.seed(seed)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.resetSpecificCurve(1, curveType = 'sine', noise = 0) # [v, omega, x, y, phi, xr, yr, phir]

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

    stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
    controlADPList = np.reshape(controlADPList, (-1, actionDim))

    stateADPList = np.delete(stateADPList, range(plotDelete), 0)
    controlADPList = np.delete(controlADPList, range(plotDelete), 0)
    rewardADP = np.delete(rewardADP, range(plotDelete), 0)

    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")


    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    timeMPCAll = []
    timeMPCAll.append(timeADP)
    for mpcstep in MPCStep:
        print("----------------------Start Solving in Real Time!----------------------")
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
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        stateMPCList = np.delete(stateMPCList, range(plotDelete), 0)
        controlMPCList = np.delete(controlMPCList, range(plotDelete), 0)
        rewardMPC = np.delete(rewardMPC, range(plotDelete), 0)

        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(simu_dir + "/simulationRealMPC_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")

        print("MPC-{} consumes {:.4f}s {} step".format(mpcstep, timeMPC, env.testStepReal))
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)
        timeMPCAll.append(timeMPC)
        # print("Overall Cost for {} Steps, MPC: {:.3f}, ADP: {:.3f}".format(env.testStepReal, rewardMPC, rewardADP.item()))

    if os.path.exists(simu_dir + "/time.csv")==False:
        with open(simu_dir + "/time.csv", 'ab') as f:
            np.savetxt(f, np.array([timeMPCAll]), delimiter=',', fmt='%.6f', comments='', header="ADP,MPC-"+",MPC-".join([str(m) for m in MPCStep]))
    else:
        with open(simu_dir + "/time.csv", 'ab') as f:
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
    ADPData =  - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    MPCData = - (stateMPCAll[-1][:,2] - stateMPCAll[-1][:,5]) * np.sin(stateMPCAll[-1][:,7]) + (stateMPCAll[-1][:,3] - stateMPCAll[-1][:,6]) * np.cos(stateMPCAll[-1][:,7])
    # ADPData = stateADPList[:, 3]
    # MPCData = stateMPCAll[-1][:, 3]
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

    if os.path.exists(simu_dir + "/RelError.csv")==False:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Steering Angle mean,max,Lateral Position mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')

    # Plot
    # stateAll = [v, omega, x, y, phi, xr, yr, phir]
    # y v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,3]
    yMPC = [mpc[:,3] for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'y position [m]'
    title = 'y-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # lateral error v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    yMPC = [- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7]) for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Lateral position error[m]'
    title = 'lateral-error-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isError = True)

    # phi v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle [°]'
    title = 'phi-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # phi error v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isError = True)

    # utility v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xName = 'Travel dist [m]'
    yName = 'utility'
    title = 'utility-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # delta v.s. x
    xADP = stateADPList[:,2]
    xMPC = [mpc[:,2] for mpc in stateMPCAll]
    yADP = controlADPList[:,0] * 180/np.pi
    yMPC = [mpc[:,0]* 180/np.pi for mpc in controlMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Steering angle [°]'
    title = 'delta-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)


def simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 0, seed = 0):
    # apply in virtual time
    plotDelete = 0
    env = TrackingEnv()
    env.seed(seed)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.resetRandom(1, noise = noise, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < max(MPCStep)):
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
    with open(simu_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")

    # MPC
    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    for mpcstep in MPCStep:
        print("----------------------Start Solving in Virtual Time!----------------------")
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
        with open(simu_dir + "/simulationVirtualMPC_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="v,omega,x,y,phi,xr,yr,phir,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)

    # Cal relative error
    errorSaveList = []
    # Steering Angle
    ADPData = controlADPList[:,0]
    MPCData = controlMPCAll[-1][:, 0]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Steering Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Lateral Position
    ADPData =  - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    MPCData = - (stateMPCAll[-1][:,2] - stateMPCAll[-1][:,5]) * np.sin(stateMPCAll[-1][:,7]) + (stateMPCAll[-1][:,3] - stateMPCAll[-1][:,6]) * np.cos(stateMPCAll[-1][:,7])
    # ADPData = stateADPList[:, 3]
    # MPCData = stateMPCAll[-1][:, 3]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Lateral Position', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Heading Angle
    ADPData = stateADPList[:,4] - stateADPList[:,7]
    MPCData = stateMPCAll[-1][:, 4] - stateMPCAll[-1][:, 7]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Heading Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Utility  Function
    ADPData = rewardADP
    MPCData = rewardMPCAll[-1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Utility  Function', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    if os.path.exists(simu_dir + "/RelError.csv")==False:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Steering Angle mean,max,Lateral Position mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')


    # Plot
    # stateAll = [v, omega, x, y, phi, xr, yr, phir]
    # delta v.s. t
    xADP = np.arange(0, len(controlADPList[:,0]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,0]) * env.T, env.T) for mpc in controlMPCAll]
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Steering angle [°]'
    title = 'delta-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi error v.s. t
    xADP = np.arange(0, len(stateADPList[:,4]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,4]) * env.T, env.T) for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # lateral error v.s. t
    yADP = - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    yMPC = [- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7]) for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'Lateral position error[m]'
    title = 'lateral-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi v.s. lateral error
    xADP = - (stateADPList[:,2] - stateADPList[:,5]) * np.sin(stateADPList[:,7]) + (stateADPList[:,3] - stateADPList[:,6]) * np.cos(stateADPList[:,7])
    xMPC = [- (mpc[:,2] - mpc[:,5]) * np.sin(mpc[:,7]) + (mpc[:,3] - mpc[:,6]) * np.cos(mpc[:,7]) for mpc in stateMPCAll]
    yADP = stateADPList[:,4] * 180/np.pi - stateADPList[:,7] * 180/np.pi
    yMPC = [mpc[:,4] * 180/np.pi - mpc[:,7] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Lateral position error [m]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-lateral-error'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # y v.s. x
    xADP = stateADPList[:, 2]
    xMPC = [mpc[:, 2] for mpc in stateMPCAll]
    yADP = stateADPList[:, 3]
    yMPC = [mpc[:, 3] for mpc in stateMPCAll]
    xName = 'Tracel dist [m]'
    yName = 'y posision [m]'
    title = 'y-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # utility v.s. t
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'utility'
    title = 'utility-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True )

def  simulationValue(MPCStep, ADP_dir, simu_dir, isLoad = False):
    env = TrackingEnv()
    relstateDim = env.relstateDim
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()

    # ADP
    state = env.resetRandom(1, noise=0) # [v, omega, x, y, phi, xr, yr, phir]
    deltay = torch.arange(-0.5, 0.5, 0.01)
    deltaphi = torch.arange(-0.2, 0.2, 0.01)
    X, Y = torch.meshgrid(deltay, deltaphi * 180/np.pi)
    valueGridADP = torch.empty_like(X)
    X = X.numpy()
    Y = Y.numpy()
    for i in range(deltay.shape[0]):
        for j in range(deltaphi.shape[0]):
            stateUse = state.clone()
            stateUse[:, 3] += deltay[i]
            stateUse[:, 4] += deltaphi[j]
            refState = env.relStateCal(stateUse)
            valueGridADP[i][j] = value(refState).detach()
    valueGridADP = valueGridADP.numpy()
    figure = plt.figure()
    ax = Axes3D(figure)
    surf = ax.plot_surface(X, Y, valueGridADP, rstride=1,cstride=1,cmap='rainbow')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    ax.set_zlabel('Value')
    # figure.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(simu_dir + '/valueADP.png')
    plt.close()
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.contour(X, Y, valueGridADP, cmap='rainbow')
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    plt.savefig(simu_dir + '/valueADPContourf.png')
    plt.close()

    # MPC
    if isLoad == False:
        deltay = deltay.numpy()
        deltaphi = deltaphi.numpy()
        valueGridMPC = np.zeros_like(X)
        for i in range(deltay.shape[0]):
            for j in range(deltaphi.shape[0]):
                stateUse = state[0].clone().tolist()
                stateUse[3] += deltay[i]
                stateUse[4] += deltaphi[j]
                stateMpc = stateUse[:5]
                refStateMpc = stateUse[5:8]
                _, control = solver.MPCSolver(stateMpc, refStateMpc, MPCStep[-1], isReal=False)
                count = 0
                gammarForward = 1
                while(count < MPCStep[-1]):
                    action = control[count].tolist()
                    reward = env.calReward(stateMpc + refStateMpc, action, MPCflag=1)
                    temp = env.vehicleDynamic(
                        stateMpc[2], stateMpc[3], stateMpc[4], env.refV, stateMpc[0], stateMpc[1], 0, action[0], MPCflag=1)
                    stateMpc[2:5] = temp[:3] # x, y, phi
                    stateMpc[:2] = temp[4:6] # v, omega
                    refStateMpc = env.refDynamicVirtual(refStateMpc, MPCflag=1)
                    valueGridMPC[i][j] += gammarForward * reward
                    gammarForward *= solver.gammar
                    count += 1
        with open(simu_dir + "/MPCvalue.csv", 'wb') as f:
            np.savetxt(f, valueGridMPC, delimiter=',', fmt='%.8f', comments='')
    elif isLoad == True:
        valueGridMPC = np.loadtxt(simu_dir + "/MPCvalue.csv", delimiter=',')
    figure = plt.figure()
    ax = Axes3D(figure)
    surf = ax.plot_surface(X, Y, valueGridMPC, rstride=1,cstride=1,cmap='rainbow')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    ax.set_zlabel('Value')
    # figure.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(simu_dir + '/valueMPC.png')
    plt.close()
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.contour(X, Y, valueGridMPC, cmap='rainbow')
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    plt.savefig(simu_dir + '/valueMPCContourf.png')
    plt.close()
    figure = plt.figure()
    ax = Axes3D(figure)
    surf = ax.plot_surface(X, Y, valueGridADP - valueGridMPC, rstride=1,cstride=1,cmap='rainbow')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    ax.set_zlabel('Value')
    # figure.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(simu_dir + '/valueCompare.png')
    figure = plt.figure()
    ax = Axes3D(figure)
    surf = ax.plot_surface(X, Y, np.abs((valueGridADP - valueGridMPC)/(valueGridMPC+0.2))*100, rstride=1,cstride=1,cmap='rainbow')
    # ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('Delta Y [m]')
    ax.set_ylabel('Delta Phi [°]')
    ax.set_zlabel('Relative Error [%]')
    # figure.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(simu_dir + '/valueCompareRelative.png')

def comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = False, isError = False):
    plt.figure()
    colorList = ['darkorange', 'green', 'blue', 'yellow', 'red']
    if isMark == True:
        markerList = ['|', 'D', 'o', 'x', '*']
    else:
        markerList = ['None', 'None', 'None', 'None']
    for i in range(len(xMPC)):
        plt.plot(xMPC[i], yMPC[i], linewidth=2, color = colorList[i], linestyle = '--', marker=markerList[i], markersize=4)
    plt.plot(xADP, yADP, linewidth = 2, color=colorList[-1],linestyle = '--', marker=markerList[-1], markersize=4)
    if isError == True:
        plt.plot([np.min(xADP), np.max(xADP)], [0,0], linewidth = 1, color = 'grey', linestyle = '--')
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP', 'Ref'])
    else:
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP'])
    plt.xlabel(xName)
    plt.ylabel(yName)
    # plt.subplots_adjust(left=)
    plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.close()

def animationPlot(state, refstate, xName, yName):
    plt.figure()
    plt.ion()
    plt.xlabel(xName)
    plt.ylabel(yName)
    colorList = ['green', 'darkorange', 'blue', 'yellow']
    plt.xlim([min(np.min(state[:,0]), np.max(refstate[:,0])), max(np.max(state[:,0]), np.max(refstate[:,0]))])
    plt.ylim([min(np.min(state[:,1]), np.max(refstate[:,1])), max(np.max(state[:,1]), np.max(refstate[:,1]))])
    for step in range(state.shape[0]):
        plt.pause(1)
        plt.scatter(state[step][0], state[step][1], color='red', s=5)
        plt.scatter(refstate[step][0], refstate[step][1], color='blue', s=5)
    plt.pause(20)
    plt.ioff()
    plt.close()

def calRelError(ADP, MPC, title, simu_dir, isPlot = False):
    maxMPC = np.max(MPC, 0)
    minMPC = np.min(MPC, 0)
    relativeError = np.abs((ADP - MPC)/(maxMPC - minMPC + 1e-3))
    relativeErrorMax = np.max(relativeError, 0)
    relativeErrorMean = np.mean(relativeError, 0)
    print(title +' Error | Mean: {:.4f}%, Max: {:.4f}%'.format(relativeErrorMean*100,relativeErrorMax*100))
    if isPlot == True:
        plt.figure()
        data = relativeError
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error of '+title)
        plt.ylabel('Frequency')
        plt.title('Relative Error of '+title)
        plt.savefig(simu_dir + '/relative-error-'+title+'.png')
        plt.close()
    return relativeErrorMean, relativeErrorMax

if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep
    # check reward
    ADP_dir = './Results_dir/2022-05-09-20-51-08'
    # 1. Apply in real time
    simu_dir = ADP_dir + '/simulationReal'
    os.makedirs(simu_dir, exist_ok=True)
    for seed in range(1):
        print('seed = {}'.format(seed))
        simulationReal(MPCStep, ADP_dir, simu_dir, seed=seed)

    # 2. Apply in virtual time
    simu_dir = ADP_dir + '/simulationVirtual'
    os.makedirs(simu_dir, exist_ok=True)
    # for seed in range(100):
    for seed in [5]:
        print('seed = {}'.format(seed))
        simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 1, seed = seed)

    # # 3. Value
    # simu_dir = ADP_dir + '/simulationValue'
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationValue(MPCStep, ADP_dir, simu_dir, isLoad = True)