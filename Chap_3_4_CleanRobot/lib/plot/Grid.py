"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

from typing import Tuple, List
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
from matplotlib import patches, animation
from matplotlib.ticker import PercentFormatter
import numpy as np

from .config import default_cfg
from ..agents import BaseGridAgent



class Grid:
    default_u = np.array([0, 1, 0, -1])
    default_v = np.array([1, 0, -1, 0])

    bbox_x = np.array([0, 1, 1, 0])
    bbox_y = np.array([0, 0, 1, 1])

    def __init__(self, grid_size: Tuple[int, int], target: Tuple[int, int], cell_size: int = 10):
        self.grid_size = grid_size
        self.target = target
        self.size = cell_size
        self.bound = (grid_size[0] * cell_size, grid_size[1] * cell_size)

    def get_center(self, row, col):
        return col * self.size + self.size / 2, row * self.size + self.size / 2

    def get_bbox(self, row, col):
        return (Grid.bbox_x + col) * self.size, (Grid.bbox_y + row) * self.size

    def plot_grid(self, ax: axes.Axes):
        ax.set_axis_off()
        ax.set_aspect('equal')
        vls = np.arange(self.grid_size[1] + 1) * self.size
        ax.vlines(vls, ymin=0, ymax=self.bound[0], lw=default_cfg.linewidth)
        hls = np.arange(self.grid_size[0] + 1) * self.size
        ax.hlines(hls, xmin=0, xmax=self.bound[1], lw=default_cfg.linewidth)
        ax.fill(*self.get_bbox(*self.target))

    def plot_policy(self, policy: np.ndarray, ax: axes.Axes):
        # self.ax.add_patch(patches.FancyArrow(25, 25, 4, 0, width=0.4, length_includes_head=True))

        quiver_x = []
        quiver_y = []
        quiver_u = []
        quiver_v = []

        delta_u = Grid.default_u * self.size / 2 * 0.9
        delta_v = Grid.default_v * self.size / 2 * 0.9

        policy = policy.reshape(self.grid_size + (-1,))
        for row in range(policy.shape[0]):
            for col in range(policy.shape[1]):
                p = policy[row, col]
                if np.abs(np.sum(p) - 1) > 1e-7:
                    continue
                large = np.argwhere(p == np.max(p))
                x, y = self.get_center(row, col)
                for idx in large:
                    u = delta_u[idx]
                    v = delta_v[idx]
                    quiver_x.append(x)
                    quiver_y.append(y)
                    quiver_u.append(u)
                    quiver_v.append(v)
        ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v)

    def plot_matrix(self, matrix: np.ndarray, ax: axes.Axes):
        matrix = matrix.reshape(self.grid_size)
        for pos, v in np.ndenumerate(matrix):
            if pos == self.target:
                continue
            ax.annotate('%.1f' % v, xy=self.get_center(*pos), ha='center', va='center', **default_cfg.label_font)

    def plot_route(self, route: List[Tuple[int, int]], ax: axes.Axes):
        for i in range(len(route) - 1):
            fr = route[i]
            to = route[i + 1]
            x, y = self.get_center(*fr)
            tx, ty = self.get_center(*to)
            dx, dy = tx - x, ty - y
            if dx != 0 or dy != 0:
                ax.add_patch(patches.FancyArrow(x, y, dx, dy, width=0.75, length_includes_head=True, edgecolor=[0, 0, 0, 0], facecolor=[0, 0, 0, 1], zorder=10))

    def plot_route_animated(self, route: List[Tuple[int, int]], fig: plt.Figure, ax: axes.Axes):
        def get_pos(idx):
            fr = route[idx]
            to = route[idx + 1]
            x, y = self.get_center(*fr)
            tx, ty = self.get_center(*to)
            dx, dy = tx - x, ty - y
            return x, y, dx, dy
        if len(route) == 0:
            return
        length = len(route)
        params = [get_pos(idx) for idx in range(length - 1)]
        artists = [[ax.add_patch(patches.FancyArrow(*param, width=0.75, length_includes_head=True, edgecolor=[0, 0, 0, 0], facecolor=[0, 0, 0, 1], zorder=10)) for param in params[:i]] for i in range(length)]
        for g in artists[1:]:
            g[-1].set_facecolor([1, 0.21, 0.21, 1])
            g[-1].set_zorder(20)
        ani = animation.ArtistAnimation(fig, artists, interval=1000, repeat=False)
        return ani

    @staticmethod
    def plot_frequency(ax: mplot3d.axes3d, matrix: np.ndarray):
        yr = np.arange(matrix.shape[1])
        xr = np.arange(matrix.shape[0])

        _xx, _yy = np.meshgrid(yr, xr)
        x, y = _xx.ravel(), _yy.ravel()

        top = matrix[tuple(y), tuple(x)] / np.sum(matrix) * 100
        bottom = np.zeros_like(top)
        width = depth = 0.6

        # credit to https://stackoverflow.com/questions/11950375/apply-color-map-to-mpl-toolkits-mplot3d-axes3d-bar3d
        offset = top + np.abs(top.min())
        fracs = offset.astype(float) / offset.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        color_values = cm.winter(norm(fracs.tolist()))

        ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=color_values, zsort='max')
        ax.w_xaxis.set_ticks(xr + 0.3)
        ax.w_xaxis.set_ticklabels(xr + 1)
        ax.w_yaxis.set_ticks(yr + 0.5)
        ax.w_yaxis.set_ticklabels(yr + 1)
        ax.w_zaxis.set_major_formatter(PercentFormatter())

        ax.set_zlabel('Frequency',fontdict=default_cfg.label_font)

    @staticmethod
    def create(agent: BaseGridAgent, *args, **kwargs):
        return Grid(grid_size=agent.grid_size, target=agent.target_position, *args, **kwargs)
