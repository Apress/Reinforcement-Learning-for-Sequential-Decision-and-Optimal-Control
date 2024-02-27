import math
import os
import time
from math import *

import gym
import matplotlib.patches as mpaches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from config import vehicleDynamic
from network import Actor, Critic


class TrackingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        config = vehicleDynamic()
        # reference velocity
        self.refV = config.refV
        self.curveK = config.curveK
        self.curveA = config.curveA
        # vehicle parameters
        self.T = config.T  # time interval
        self.m = config.m  # mass
        self.a = config.a  # the center of mass to the front axis
        self.b = config.b  #  the center of mass to the rear axis
        self.kf = config.kf  # total lateral stiffness of front wheels
        self.kr = config.kr  # total lateral stiffness of rear wheels
        self.Iz = config.Iz  # rotational inertia

        self.initState = config.initState
        self.testStepReal = config.testStepReal
        self.testStepVirtual = config.testStepVirtual
        self.testSampleNum = config.testSampleNum
        self.renderStep = config.renderStep

        # action space
        # u = [delta]
        # If you modify the range, you must modify the output range of Actor.
        self.actionLow = [-0.3]
        self.actionHigh = [0.3]
        self.actionSpace = \
            spaces.Box(low=np.array(self.actionLow),
                       high=np.array(self.actionHigh), dtype=np.float64)

        # state space
        # x = [v, omega, x, y, phi]
        # v is the lateral velocities
        self.stateLow = [-5*self.refV, -20, -inf, -inf, -inf]
        self.stateHigh = [5*self.refV, 20, inf, inf, inf]
        self.stateDim = 8 # augmented state dimensions, \bar x = [v, omega, x, y, phi, xr, yr, phir]
        self.relstateDim = 5 # relative state, input of NN, x_r = [v, omega, ye, cos(phie), sin(phie)]
    

    def seed(self, s):
        # random seed
        np.random.seed(s)
        torch.manual_seed(s)


    def resetRandom(self, stateNum, noise = 1, MPCflag = 0, MPCtest=False):
        # augmented state space \bar x = [v, omega, x, y, phi, xr, yr, phir]
        newState = torch.empty([stateNum, self.stateDim])
        # v: [-self.refV/8, self.refV/8]
        newState[:, 0] = 2 * (torch.rand(stateNum) - 1/2) * self.refV / 8 * noise
        # omega: [-1, 1]
        newState[:, 1] = 2 * (torch.rand(stateNum) - 1/2) * noise
        # x
        newState[:, 2] = torch.zeros((stateNum))
        # y: [-self.refV * self.T, self.refV * self.T] * 1.5
        newState[:, 3] = 2 * (torch.rand(stateNum) - 1/2) * self.refV * self.T * 1.5 * noise
        # phi: [-pi/10, pi/10] [-18, 18]
        newState[:, 4] = 2 * (torch.rand(stateNum) - 1/2) * np.pi/10 * noise
        # [xr, yr, phir]
        newState[:, 5:] = torch.zeros((stateNum,3))
        if MPCflag == 0:
            return newState
        else:
            return newState[0].tolist()
         

    def resetSpecificCurve(self, stateNum, curveType = 'sine', noise = 0):
        # augmented state space \bar x = [v, omega, x, y, phi, xr, yr, phir]
        newState = torch.empty([stateNum, self.stateDim])
        newState[:, 0] = torch.zeros(stateNum) # v
        newState[:, 1] = torch.zeros(stateNum) # omega
        newState[:, 2] = torch.zeros(stateNum) # x
        newState[:, 3], newState[:, 4] = self.referenceCurve(newState[:, 2], curveType = curveType) # y, phi
        newState[:, 5] = newState[:, 2] + self.refV * self.T # xr
        newState[:, 6], newState[:, 7] = self.referenceCurve(newState[:, 5], curveType = curveType) # yr, phir
        return newState


    def stepReal(self, state, control, curveType = 'sine', noise = 0):
        # You must initialize all state for specific curce!
        # step in real time
        # \bar x = [v, omega, x, y, phi, xr, yr, phir]
        newState = torch.empty_like(state)
        # input of vehicleDynamic: x_0, y_0, phi_0, u_0, v_0, omega_0, acc, delta
        temp = \
            torch.stack(self.vehicleDynamic(state[:, 2], state[:, 3], state[:, 4], self.refV, state[:, 0],
                                            state[:, 1], torch.zeros(state.shape[0]), control[:, 0]), -1)
        newState[:, 2:5] = temp[:, :3] # x, y, phi
        newState[:, :2] = temp[:, 4:6] # v, omega
        # TODO: you can add some specific trajectory here
        newState[:, 5:8] = self.refDynamicReal(state[:, 5:8], curveType = curveType, noise = noise)
        reward = self.calReward(state, control)  # calculate using current state
        done = self.isDone(newState, control)
        return newState, reward, done


    def stepVirtual(self, state, control):
        newState = torch.empty_like(state)
        temp = \
            torch.stack(self.vehicleDynamic(state[:, 2], state[:, 3], state[:, 4], self.refV, state[:, 0],
                                            state[:, 1], torch.zeros(state.size(0)), control[:, 0]), -1)
        # temp = [x, y, phi, u, v, omega]
        # state = [v, omega, x, y, phi, xr, yr, phir]
        newState[:, 2:5] = temp[:, :3] # x, y, phi
        newState[:, :2] = temp[:, 4:6] # v, omega
        newState[:, 5:8] = self.refDynamicVirtual(state[:, 5:8])
        reward = self.calReward(state, control)
        done = self.isDone(newState, control)
        return newState, reward, done


    def calReward(self, state, control, MPCflag = 0):
        # TODO: design reward
        if MPCflag == 0:
            deltaL = -(state[:, 2] - state[:, 5]) * torch.sin(state[:, 7]) + (state[:, 3] - state[:, 6]) * torch.cos(state[:, 7])
            deltaphi = state[:, 4] - state[:, 7]
            reward = torch.pow(deltaL, 2) + 10 * torch.pow(deltaphi, 2) + 10 * torch.pow(control[:, 0], 2)
        else:
            return self.calReward(torch.tensor([state]), torch.tensor([control]), MPCflag=0)[0].tolist()
        return reward


    def isDone(self, state, control):
        # TODO: design condition of done
        # e.x. np.pi/6->np.pi/10
        batchSize = state.size(0)
        done = torch.tensor([False for i in range(batchSize)])
        deltaL = -(state[:, 2] - state[:, 5]) * torch.sin(state[:, 7]) + (state[:, 3] - state[:, 6]) * torch.cos(state[:, 7])
        deltaphi = state[:, 4] - state[:, 7]
        done[torch.abs(deltaL) > 4] = True
        done[torch.abs(deltaphi) > np.pi/6] = True
        return done


    def vehicleDynamic(self, x_0, y_0, phi_0, u_0, v_0, omega_0, acc, delta, MPCflag = 0):
        if MPCflag == 0:
            x_1 = x_0 + self.T * (u_0 * torch.cos(phi_0) - v_0 * torch.sin(phi_0))
            y_1 = y_0 + self.T * (v_0 * torch.cos(phi_0) + u_0 * torch.sin(phi_0))
            phi_1 = phi_0 + self.T * omega_0
            u_1 = u_0 + self.T * acc
            v_1 = (-(self.a * self.kf - self.b * self.kr) * omega_0 + self.kf * delta * u_0 +
                self.m * omega_0 * u_0 * u_0 - self.m * u_0 * v_0 / self.T) \
                / (self.kf + self.kr - self.m * u_0 / self.T)
            omega_1 = (-self.Iz * omega_0 * u_0 / self.T - (self.a * self.kf - self.b * self.kr) * v_0
                    + self.a * self.kf * delta * u_0) \
                / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * u_0 / self.T)
        else:
            x_1 = x_0 + self.T * (u_0 * cos(phi_0) - v_0 * sin(phi_0))
            y_1 = y_0 + self.T * (v_0 * cos(phi_0) + u_0 * sin(phi_0))
            phi_1 = phi_0 + self.T * omega_0
            u_1 = u_0 + self.T * acc
            v_1 = (-(self.a * self.kf - self.b * self.kr) * omega_0 + self.kf * delta * u_0 +
                self.m * omega_0 * u_0 * u_0 - self.m * u_0 * v_0 / self.T) \
                / (self.kf + self.kr - self.m * u_0 / self.T)
            omega_1 = (-self.Iz * omega_0 * u_0 / self.T - (self.a * self.kf - self.b * self.kr) * v_0
                    + self.a * self.kf * delta * u_0) \
                / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * u_0 / self.T)
        return [x_1, y_1, phi_1, u_1, v_1, omega_1]


    def refDynamicVirtual(self, refState, MPCflag = 0):
        if MPCflag == 0:
            newRefState = torch.empty_like(refState)
            newRefState = refState
        else:
            return self.refDynamicVirtual(torch.tensor([refState]), MPCflag = 0)[0].tolist()
        return newRefState


    def refDynamicReal(self, refState, MPCflag = 0, curveType = 'sine', noise = 0):
        if MPCflag == 0:
            newRefState = torch.empty_like(refState) # [xr, yr, phir]
            newRefState[:, 0] = refState[:, 0] + self.T * self.refV
            newRefState[:, 1], newRefState[:, 2] = self.referenceCurve(newRefState[:, 0], curveType = curveType, MPCflag = MPCflag)
        else:
            return self.refDynamicReal(torch.tensor([refState]), MPCflag = 0, curveType = curveType, noise = noise)[0].tolist()
        return newRefState


    def referenceCurve(self, x, MPCflag = 0,  curveType = 'sine'):
        if MPCflag == 0:
            if curveType == 'sine':
                return self.curveA * torch.sin(self.curveK * x), torch.atan(self.curveA * self.curveK * torch.cos(self.curveK * x))
        else:
            if curveType == 'sine':
                return self.curveA * sin(self.curveK * x), atan(self.curveA * self.curveK * cos(self.curveK * x))


    def relStateCal(self, state):
        # \bar x = [v, omega, x, y, phi, xr, yr, phir]
        # relState = [v, omega, dL, dphi]
        batchSize = state.size(0)
        relState = torch.empty([batchSize, self.relstateDim])
        relState[:, :2] = state[:, :2]
        relState[:, 2] = -(state[:, 2] - state[:, 5]) * torch.sin(state[:, 7]) + (state[:, 3] - state[:, 6]) * torch.cos(state[:, 7])
        deltaphi = state[:, 4] - state[:, 7]
        relState[:, 3] = torch.cos(deltaphi)
        relState[:, 4] = torch.sin(deltaphi)
        return relState


    def policyTestReal(self, policy, iteration, log_dir, curveType = 'sine', noise = 0):
        state  = self.resetSpecificCurve(1, curveType = curveType)
        count = 0
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        rewardSum = 0

        reversalList = [40, self.testStepReal-40]

        while(count < self.testStepReal):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())

            # if count < reversalList[0] or count > reversalList[1]:
            #     state, reward, done = self.stepReal(state, control, curveType=curveType, noise = 0)
            # else:
            state, reward, done = self.stepReal(state, control, curveType=curveType, noise = noise)

            rewardSum += min(reward.item(), 100000/self.testStepReal)
            count += 1
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        controlADP = np.reshape(controlADP, (-1, 1))
        saveADP = np.concatenate((stateADP, controlADP), 1) # [v, omega, x, y, phi, xr, yr, phir, delta]
        # with open(log_dir + "/Real_state"+str(iteration)+".csv", 'wb') as f:
        with open(log_dir + "/Real_last_state.csv", 'wb') as f:
            np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v, omega, x, y, phi, xr, yr, phir, delta")
        plt.figure()
        plt.scatter(stateADP[:, 2], stateADP[:, 3], color='red', s=0.5)
        plt.scatter(stateADP[:, 5], stateADP[:, 6], color='gray', s=0.5)
        # plt.scatter(stateADP[:, -3], stateADP[:, -2],  s=20, c='red', marker='*')
        # plt.scatter(stateADP[:, 3], stateADP[:, 4], c='gray', s = 20, marker='+')
        plt.legend(labels = ['ADP', 'reference'])
        # plt.axis('equal')
        plt.title('iteration:'+str(iteration))
        # plt.savefig(log_dir + '/Real_iteration'+str(iteration)+'.png')
        plt.savefig(log_dir + '/Real_last_iteration.png')
        plt.close()
        return rewardSum

    def policyTestVirtual(self, policy, iteration, log_dir, noise = 0, isPlot=True):
        if isPlot == True:
            state = self.resetRandom(1, noise=noise)
        else:
            state = self.resetRandom(self.testSampleNum, noise=noise)
        count = 0
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        rewardSum = 0
        while(count < self.testStepVirtual):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())
            state, reward, done = self.stepVirtual(state, control)
            rewardSum += torch.mean(torch.min(reward,torch.tensor(50))).item()
            count += 1
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        controlADP = np.reshape(controlADP, (-1, 1))
        saveADP = np.concatenate((stateADP, controlADP), 1) # [x, y, phi, v, omega, xr, yr, phir, delta]
        if isPlot == True:
            # with open(log_dir + "/Virtual_state"+str(iteration)+".csv", 'wb') as f:
            with open(log_dir + "/Virtual_last_state.csv", 'wb') as f:
                np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="v, omega, x, y, phi, xr, yr, phir, delta")
            plt.figure()
            plt.scatter(stateADP[:, 2], stateADP[:, 3],  s=20, c='red', marker='*')
            plt.scatter(stateADP[:, 5], stateADP[:, 6], c='gray', s = 20, marker='+')
            plt.legend(labels = ['ADP', 'reference'])
            # plt.axis('equal')
            plt.title('iteration:'+str(iteration))
            # plt.savefig(log_dir + '/Virtual_iteration'+str(iteration)+'.png')
            plt.savefig(log_dir + '/Virtual_last_iteration.png')
            plt.close()
        return rewardSum


    def plotReward(self, rewardSum, log_dir, saveIteration):
        plt.figure()
        plt.plot(range(0,len(rewardSum)*saveIteration, saveIteration),rewardSum)
        plt.xlabel('itetation')
        plt.ylabel('reward')
        plt.savefig(log_dir + '/reward.png')
        plt.close()

if __name__ == '__main__':
    ADP_dir = './Results_dir/2022-04-10-23-28-38'
    log_dir = ADP_dir + '/test'
    os.makedirs(log_dir, exist_ok=True)
    env = TrackingEnv()
    # env.seed(0)

    policy = Actor(env.relstateDim, env.actionSpace.shape[0])
    policy.loadParameters(ADP_dir)
    # env.policyRender(policy)
    noise = 0.25
    env.curveK = 1/10
    env.curveA = 1
    env.policyTestReal(policy, 0, log_dir, curveType = 'sine', noise = noise)
    # env.policyTestReal(policy, 4, log_dir, curveType = 'sine', noise = 0)
    env.policyTestVirtual(policy, 0, log_dir, noise = 1)


