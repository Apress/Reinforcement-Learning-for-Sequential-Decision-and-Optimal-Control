import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import scipy.signal
from lib.agent import StochasticPolicy, DeterminePolicy

MAX_STEP_EP = 400  # maximum step the agent can run in the environment

def smooth(data, a=1):
    data = np.array(data).reshape(-1, 1)
    last = data[0, 0]
    for ind in range(data.shape[0] - 1):
        now = data[ind + 1, 0]
        data[ind + 1, 0] = last * (1-a) + data[ind + 1, 0] * a
        last = now
    return data

def aggre(data, step=1):
    data_aggre = [data[i] for i in range(0,len(data),step)]
    return data_aggre

def discount(x, gamma):
    """
    Compute discounted sum of future values. Returns a list, NOT a scalar!
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def policy_evaluate(env, actor, mode):
    """
    evaluate the policy by Total average return

    """
    actor.eval()
    returns = []
    start = []
    # sample different s0 from initial distribution
    for i in range(10):
        s0 = env.reset()
        start.append(s0)
        for _ in range(5):
            s = env.reset(init_state=s0)
            rewards = []
            time_step_ep = 0
            while True:
                if mode == "sto":
                    a, _ = actor.choose_action(s)
                elif mode == "det":
                    _, a = actor.choose_action(s)
                s, r, done, mask = env.step(a[0])
                rewards.append(r)
                time_step_ep += 1
                if done or time_step_ep >= MAX_STEP_EP:
                    break
            returns.append(np.sum(rewards))
    tar = np.mean(returns)
    actor.train()
    return tar, start

def myplot_var(x,
               y_mean,
               y_var,
                color_list,
           fname=None,
           xlabel=None,
           ylabel=None,
           legend=None,
           legend_loc="best",

           xlim=None,
           ylim=None,
           yline=None,
           xline=None,
           ncol=1):
    """
    plot figures
    """

    _, ax = plt.subplots()

    if color_list is not None:
        for i, (x1, y1, y2) in enumerate(zip(x, y_mean, y_var)):
            plt.plot(x1, y1, color=color_list[i])
            r1 = list(map(lambda x: x[0] - x[1], zip(y1, y2)))
            r2 = list(map(lambda x: x[0] + x[1], zip(y1, y2)))
            ax.fill_between(x1, r1, r2, color=color_list[i], alpha=0.2)

    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Calibri') for label in labels]
    font = {'family': 'Calibri', 'size': '18'}
    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=font)
        # 'lower center'
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)