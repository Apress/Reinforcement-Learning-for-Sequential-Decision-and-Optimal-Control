"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

"""

import numpy as np

roadColumn = 24  # Column number of road cells
roadRow = 12  # Row number of road cells
carDirection = 5  # Car carDirection (1-head up, 5- head down)
state_size = [roadColumn, roadRow, carDirection]  # X of road, Y of road, Heading of car
action_size = 3  # left;right;keep
laneBoundary = np.array([
    [12, 12, 12, 11, 11, 10, 9, 8, 7, 7, 6, 6, 6, 7, 7, 8, 9, 10, 11, 11, 12, 12, 12, 12],
    [7, 7, 7, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 7, 7]
]) - 1  # Upper and lower boundary of curved road
s_a = [roadColumn, roadRow, carDirection, action_size]
gridNum = sum(laneBoundary[0, :] - laneBoundary[1, :] + 1)