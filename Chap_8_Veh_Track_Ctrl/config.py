"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 8: Simulate MPC controller and ADP controller for comparison

"""

import math

class trainConfig():
    def __init__(self):
        self.iterationMax = 1000
        self.iterationPrint = 10
        self.iterationSave = 1000
        self.lrPolicy = 1e-3
        self.lrValue = 2e-3
        self.stepForwardPEV = 80
        self.gammar = 1.00
        self.lifeMax = 20
        self.batchSize = 256


class vehicleDynamic():
    def __init__(self):
        # 参考速度
        self.refV = 5
        self.curveK = 1/10
        self.curveA = 1

        self.T = 0.02  # 时间间隔
        self.m = 1520  # 自车质量
        self.a = 1.19  # 质心到前轴的距离
        self.b = 1.46  # 质心到后轴的距离
        self.kf = -155495  # 前轮总侧偏刚度
        self.kr = -155495  # 后轮总侧偏刚度
        self.Iz = 2642  # 转动惯量

        # 初始状态
        self.initState = [0, 0, math.atan(self.curveA * self.curveK), self.refV, 0, 0]
        self.testStepReal = 1000
        self.testStepVirtual = 300
        self.testSampleNum = 100
        self.renderStep = 100

class MPCConfig():
    def __init__(self):
        self.MPCStep = [20, 60, 150]
        config = trainConfig()
        self.gammar = config.gammar

class PlotConfig(object):
    fig_size = (8.5, 6.5)
    dpi = 300
    pad = 0.2
    tick_size = 8
    legend_font = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
    label_font = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}
    tick_label_font = 'Times New Roman'