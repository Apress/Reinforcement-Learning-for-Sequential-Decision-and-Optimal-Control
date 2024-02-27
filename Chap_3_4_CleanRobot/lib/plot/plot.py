"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang

Description: Chapter 3 & 4: 

"""

from matplotlib import pyplot as plt
import numpy as np

from .Grid import Grid
from .config import default_cfg


def plot_policy(agent, index, ax):
    policy = agent.policy if index == -1 else agent.params.sampler.data['policy'][..., index]
    gr = Grid.create(agent)
    gr.plot_grid(ax)
    gr.plot_policy(policy, ax)


def plot_value(agent, index, ax):
    S_value = agent.S_value if index == -1 else agent.params.sampler.data['S_value'][..., index]
    gr = Grid.create(agent)
    gr.plot_grid(ax)
    gr.plot_matrix(S_value, ax)


def plot_route(agent, index, ax):
    def state_to_position(r):
        return list(zip(*agent.state_to_position(r)))
    route = agent.params.sampler.data['route'][index]
    gr = Grid.create(agent)
    gr.plot_grid(ax)
    gr.plot_route(state_to_position(route), ax)


def plot_route_animated(agent, index, fig, ax):
    def state_to_position(r):
        return list(zip(*agent.state_to_position(r)))

    route = agent.params.sampler.data['route'][index]
    gr = Grid.create(agent)
    gr.plot_grid(ax)
    return gr.plot_route_animated(state_to_position(route), fig, ax)


def plot_statistics(agent, index, ax):
    action_statistics = agent.params.sampler.data['action_statistics'][..., index]
    gr = Grid.create(agent)
    gr.plot_frequency(ax, action_statistics.reshape(agent.grid_size))


def plot_rms(agents_group, dp_agent, template, values, ax):
    def get_rms(agent):
        av = agent.params.sampler.data['S_value']
        tv = dp_agent.S_value.reshape((-1, 1))
        return np.linalg.norm(av - tv, axis=0) / np.sqrt(agent.grid_area - 1)

    for i, agents in enumerate(agents_group):
        all_rms = np.array([get_rms(agent) for agent in agents])
        mean = np.mean(all_rms, axis=0)
        std = np.std(all_rms, axis=0)
        # up_lim = np.max(all_rms, axis=0)
        # lo_lim = np.min(all_rms, axis=0)

        # error region
        ep = agents[0].params.sampler.episode['S_value']
        ax.plot(ep, mean, label=template % values[i])
        ax.fill_between(ep, mean - std, mean + std, alpha=0.5, antialiased=True)

        # # error bar mode
        # ep = agents[i].params.sampler.episode['S_value']
        # error_every = 1 if ep[2] - ep[1] > 1 else 10
        # ax.errorbar(ep, mean, yerr=[mean - lo_lim, up_lim - mean], label=template % values[i], errorevery=error_every)

        # # plain mode
        # ax.plot(agents[0].params.sampler.episode['S_value'], mean, label=template % values[i])

    ax.set_xlabel('Episodes', default_cfg.label_font)
    ax.set_ylabel('RMS Error', default_cfg.label_font)
    ax.legend()


def plot_reward_calc(agents, max_value, template, values, ax, repeat=5):
    """ Time consuming. Not used. """
    from lib.agents.BaseGridAgent import BaseGridAgent
    from lib.RLParams import RLParams
    bga = BaseGridAgent(grid_size=6, rl_params=RLParams(name="BGA", max_step_per_episode=50, reward=(-1, 9)))

    def traverse_policy(policy):
        bga.policy = policy
        return bga.traverse(repeat=5)

    ax.hlines(max_value, linestyles='--', xmin=0, xmax=agents[0].params.sampler.episode['policy'][-1], label='Max of return', color='r', zorder=10)

    for i, agent in enumerate(agents):
        policies = agent.params.sampler.data['policy']
        episodes = agent.params.sampler.episode['policy']
        rewards = [traverse_policy(policies[..., i]) for i in range(policies.shape[-1])]
        ax.plot(episodes, rewards, label=(template % values[i]) if template is not None else None)

    ax.set_xlabel('Episodes', default_cfg.label_font)
    ax.set_ylabel('Total Average Return', default_cfg.label_font)
    ax.legend()


def plot_reward_pre(agents_group, rewards, dp_rewards, template, values, ax):
    for i, agents in enumerate(agents_group):
        episode = agents[0].params.sampler.episode['policy']
        mean, std, up, lo = rewards[i]
        ax.plot(episode, mean, label=template % values[i])
        ax.fill_between(episode, np.clip(mean - std, lo, mean), np.clip(mean + std, mean, up), alpha=0.5, antialiased=True)

    ax.hlines(dp_rewards, linestyles='--', xmin=0, xmax=agents_group[0][0].params.sampler.episode['policy'][-1], label='Max of return', color='r', zorder=10)
    ax.set_xlabel('Episodes', default_cfg.label_font)
    ax.set_ylabel('Total Average Return', default_cfg.label_font)
    ax.legend()

def plot_route_length(agent, ax):
    route_length = agent.params.sampler.data['route_length']
    route_cumsum = np.cumsum(route_length)
    ax.plot(range(len(route_length)), route_cumsum)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Samples")

def plot_route_length_hist(agent, ax):
    route_length = agent.params.sampler.data['route_length']
    weights = np.ones(len(route_length)) / len(route_length)
    ax.hist(route_length, weights=weights)
    ax.set_xlabel("Length of episode")
    ax.set_ylabel("Occurrence Freq")

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)