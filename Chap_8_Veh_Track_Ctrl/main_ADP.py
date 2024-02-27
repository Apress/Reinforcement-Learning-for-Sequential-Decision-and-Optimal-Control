"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 8: RL example for lane keeping problem in a curve road;
             Approximate dynamic programming with structured policy

"""


# =================== load package ====================
import os
import shutil
import time
from datetime import datetime

import torch

from config import trainConfig
from dynamics import TrackingEnv
from network import Actor, Critic
from train import Train
import main_simuMPC


# ============= Setting hyper-parameters ===============
# mode setting
isTrain = True

# parameters setting
config = trainConfig()
env = TrackingEnv()
env.seed(0)

use_gpu = torch.cuda.is_available()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = Actor(relstateDim, actionDim, lr=config.lrPolicy)
value = Critic(relstateDim, 1, lr=config.lrValue)
log_dir = "./Results/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%2d")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir + '/train', exist_ok=True)


# ================== Training =====================

if isTrain:
    print("********************************** START TRAINING **********************************")
    print("************************** PRINT LOSS EVERY " + str(config.iterationPrint) + "iterations ***************************")
    train = Train(env)
    iterarion = 0
    lossListValue = 0
    timeBegin = time.time()
    while iterarion < config.iterationMax:
        # PEV
        train.policyEvaluate(policy, value)
        # PIM
        train.policyImprove(policy, value)
        train.calLoss(iterarion)
        # update
        train.update(policy)
        if iterarion % config.iterationPrint == 0:
            print("iteration: {}, LossValue: {:.4f}, LossPolicy: {:.4f}".format(
                iterarion, train.lossValue[-1], train.lossPolicy[-1], value.opt.param_groups[0]['lr'], policy.opt.param_groups[0]['lr']))
        if iterarion % config.iterationSave == 0 or iterarion == config.iterationMax - 1:
            env.policyTestReal(policy, iterarion, log_dir+'/train')
            env.policyTestVirtual(policy, iterarion, log_dir+'/train', noise = 1)
            rewardSum = env.policyTestVirtual(policy, iterarion, log_dir+'/train', noise = 1, isPlot = False)
            timeDelta = time.time() - timeBegin
            value.saveParameters(log_dir)
            policy.saveParameters(log_dir)
            train.saveDate(log_dir+'/train')
        iterarion += 1

print("********************************* START SIMULATION *********************************")
main_simuMPC.main(log_dir)
