"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Yuxuan Jiang

Description: Chapter 3 & 4: RL example for clearning robot in grid world
             incl. Monte Carlo RL, Dynamic Programming, SARSA, Q-learning

"""


# =================== load package ====================
from matplotlib import pyplot as plt
from lib.agents.BaseGridAgent import BaseGridAgent
from lib import RLParams
from lib.agents import DPGridAgent, QLGridAgent, SarsaGridAgent, MCGridAgent
from lib.util import save_agent, load_agent
from lib.plot import plot_policy, plot_value, plot_reward_calc, plot_route_length, plot_route_length_hist, cm2inch
from lib.plot.config import default_cfg as PLT_CONF


# ================= Select Option to Run ================
option = 4   # 1=DP, 2=MC, 3=Sarsa, 4=Q-learn
render = True  # True / False


# =================== Dynamic Programming ====================
def example_dynamic_programming():
    params = RLParams(name="DP", gamma=0.95, DP_max_iteration=100,
                      DP_error_threshold=1e-8, enable_sampling=True)
    grid_agent = DPGridAgent(grid_size=6, rl_params=params)
    grid_agent.train()
    return grid_agent

# =================== Monte Carlo ====================
def example_monte_carlo():
    params = RLParams(name="MC", gamma=0.95, epsilon=0.05, episodes_per_iter=100, MC_max_iteration=20, lamda=0.5,
                      max_step_per_episode=64, MC_with_Q_initialization=True, MC_Q_error_threshold=0.01,
                      enable_sampling=True)
    grid_agent = MCGridAgent(grid_size=6, rl_params=params)
    grid_agent.train()
    return grid_agent

# =================== SARSA ====================
def example_sarsa():
    params = RLParams(name="Sarsa", gamma=0.95, alpha=0.1, epsilon=0.1, max_episodes=2000,
                      max_step_per_episode=64, Sarsa_PEV_steps=4, enable_sampling=True)
    grid_agent = SarsaGridAgent(grid_size=6, rl_params=params)
    grid_agent.train()
    return grid_agent

# =================== Q-learning ====================
def example_Q_learning():
    params = RLParams(name="QL", gamma=0.95, alpha=0.1, max_episodes=2000,
                      max_step_per_episode=64, QL_init_coefficient=1, enable_sampling=True)
    grid_agent = QLGridAgent(grid_size=(6, 6), rl_params=params)
    grid_agent.train()
    return grid_agent

def render_agent(agent: BaseGridAgent):
    fig_size_sq = (PLT_CONF.fig_size_squre * PLT_CONF.figsize_scalar, PLT_CONF.fig_size_squre * PLT_CONF.figsize_scalar)
    fig_size = (PLT_CONF.fig_size * PLT_CONF.figsize_scalar, PLT_CONF.fig_size * PLT_CONF.figsize_scalar)
    
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
    plot_policy(agent, -1, ax)
    ax.set_position([0, 0, 1, 1])
    fig.show()

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
    plot_value(agent, -1, ax)
    ax.set_position([0, 0, 1, 1])
    fig.show()

    if not isinstance(agent, DPGridAgent):
        # Sample per episode
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
        plot_route_length(agent, ax)
        plt.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
        plot_route_length_hist(agent, ax)
        plt.tight_layout()
        fig.show()

    # NOTE: Best Return = 3.366 from DP
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_reward_calc([agent], 3.366, None, None, ax, repeat=100)
    plt.tick_params(labelsize=PLT_CONF.tick_size)
    plt.ylim((-40, 5))
    if isinstance(agent, DPGridAgent):
        ax.set_xlabel("Iterations")
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(PLT_CONF.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=PLT_CONF.legend_font)
    plt.tight_layout()
    fig.show()

    plt.show()

# ================= Run ================
if option == 1:
    agent = example_dynamic_programming()
if option == 2:
    agent = example_monte_carlo()
if option == 3:
    agent = example_sarsa()
if option == 4:
    agent = example_Q_learning()

save_agent(agent, filename=agent.params.name + '_')

print(agent.run(start_position=(0, 0)))  # Test trained performance
print(agent.get_dominant_policy())  # View policy
print('End of Running!')
if render:
    render_agent(agent)
