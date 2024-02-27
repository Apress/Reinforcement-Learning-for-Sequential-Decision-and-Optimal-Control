import numpy as np
import torch
from config import trainConfig
import os
import matplotlib.pyplot as plt

class Train():
    def __init__(self, env):
        self.env = env
        self.lossIteraValue = np.empty(0)
        self.lossIteraPolicy = np.empty(0)
        self.lossValue = np.empty(0)
        self.lossPolicy = np.empty(0)
        config = trainConfig()
        self.stepForwardPEV = config.stepForwardPEV
        self.batchSize = config.batchSize
        self.alpha = 0.99
        self.gammar = config.gammar * self.alpha
        self.lifeMax = config.lifeMax
        self.statelifeMax = torch.rand(self.batchSize) * config.lifeMax
        self.batchData = torch.empty([self.batchSize, self.env.stateDim])
        self.batchDataLife = torch.zeros(self.batchSize)
        self.accumulateReward = None
        self.stateForwardNext = None
        self.doneForward = None
        self.gammarForward = 1
        self.reset()

    def reset(self):
        self.batchData = self.env.resetRandom(self.batchSize)

    def update(self, policy):
        relState = self.env.relStateCal(self.batchData)
        control = policy(relState).detach()
        self.batchData, _, done = self.env.stepVirtual(self.batchData, control)
        self.batchDataLife += 1
        if sum(done==True) >0 :
            self.batchData[done==1] = self.env.resetRandom(sum(done==1))
            self.batchDataLife[done==1] = 0
        if sum(self.batchDataLife > self.statelifeMax) > 0:
            temp = (self.batchDataLife > self.statelifeMax)
            self.batchData[temp] =self.env.resetRandom(sum(temp))
            self.batchDataLife[temp] = 0
            self.statelifeMax[temp] = torch.rand(sum(temp)) * self.lifeMax

    def policyEvaluate(self, policy, value):
        relState = self.env.relStateCal(self.batchData)
        valuePredict = value(relState)
        valueTaeget = torch.zeros(self.batchSize)
        stateNext = self.batchData.clone()
        self.gammarForward = 1
        for _ in range(self.stepForwardPEV):
            relState = self.env.relStateCal(stateNext)
            control = policy(relState)
            stateNext, reward, done = self.env.stepVirtual(stateNext, control)
            valueTaeget += reward * self.gammarForward * (~done)
            self.gammarForward *= self.gammar
        self.accumulateReward = valueTaeget.clone()
        self.stateForwardNext = stateNext.clone()
        self.doneForward = done
        relState = self.env.relStateCal(stateNext)
        valueTaeget += (~done) * value(relState) * self.gammarForward
        valueTaeget = valueTaeget.detach()
        lossValue = torch.pow(valuePredict - valueTaeget, 2).mean()
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossIteraValue = np.append(
            self.lossIteraValue, lossValue.detach().cpu().numpy())

    def policyImprove(self, policy, value):
        for p in value.parameters():
            p.requires_grad = False
        relState = self.env.relStateCal(self.stateForwardNext)
        valueTarget = self.accumulateReward + (~self.doneForward) * value(relState) * self.gammarForward
        for p in value.parameters():
            p.requires_grad = True
        policy.zero_grad()
        lossPolicy = valueTarget.mean()
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossIteraPolicy = np.append(
            self.lossIteraPolicy, lossPolicy.detach().cpu().numpy())

    def calLoss(self, iter):
        self.lossValue = np.append(self.lossValue, self.lossIteraValue.mean())
        self.lossPolicy = np.append(self.lossPolicy, self.lossIteraPolicy.mean())
        self.lossIteraValue = np.empty(0)
        self.lossIteraPolicy = np.empty(0)

    def saveDate(self, log_dir):
        # TODO: loss 数量不一样
        with open(log_dir + "/loss.csv", 'wb') as f:
            np.savetxt(f, np.stack((self.lossValue, self.lossPolicy), 1), delimiter=',', fmt='%.4f', comments='', header="valueLoss,policyLoss")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(range(len(self.lossValue)), self.lossValue, color = 'blue', label = 'Value Loss')
        ax.set_xlabel('iteration')
        ax.set_ylabel('Value Loss')
        ax2 = ax.twinx()
        ax2.plot(range(len(self.lossPolicy)), self.lossPolicy, color = 'orange', label = 'Policy Loss')
        ax2.set_ylabel('Policy Loss')
        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.savefig(log_dir + '/loss.png', bbox_inches='tight')
        plt.close()