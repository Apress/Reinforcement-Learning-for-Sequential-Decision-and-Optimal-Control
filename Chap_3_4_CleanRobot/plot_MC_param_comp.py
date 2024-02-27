"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Yuxuan Jiang

Description: Chapter 3 & 4: Plot figures for DP, MC, Sarsa, and Q-learning algorithms

"""

from common_plot import *


# ================ Select Option ================
LOAD_PRETRAINED = 2
# 0 - calculate both;
# 1 - load agents, calculate returns;
# 2 - load both agents and returns.
REPEAT_NUM = 10
OUTPUT_DIR = OUTPUT_ROOT_DIR + "/MC_param"
DIR_NAME = DIR_ROOT_NAME + "/MC_param"

# ================ Prepare all agents for plotting ================
def prepare_agents():
    def dp():
        params_ = RLParams(name="DP", gamma=0.95, DP_max_iteration=100,
                           DP_error_threshold=1e-8)
        dp_grid_agent_ = DPGridAgent(**common_problem_defs, rl_params=params_)
        dp_grid_agent_.train()
        return dp_grid_agent_

    def mc_wo_init():
        params_ = RLParams(name="MC w/o init",
                           gamma=0.95, epsilon=0.1, lamda=0.25, episodes_per_iter=100, MC_max_iteration=20,
                           max_step_per_episode=64, MC_with_Q_initialization=False, enable_sampling=True)
        mc_wo_init_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_wo_init_grid_agent_.train()
        return mc_wo_init_grid_agent_

    def mc_with_init():
        params_ = RLParams(name="MC w/ init",
                           gamma=0.95, epsilon=0.1, lamda=0.25, episodes_per_iter=100, MC_max_iteration=20,
                           max_step_per_episode=64, MC_with_Q_initialization=True, MC_Q_error_threshold=0,
                           enable_sampling=True)
        mc_with_init_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_with_init_grid_agent_.train()
        return mc_with_init_grid_agent_

    @repeat(REPEAT_NUM)
    def mc_pev(ep_per_pev: int):
        params_ = RLParams(name=f"MC ep/pev={ep_per_pev}",
                           gamma=0.95, epsilon=0.1, lamda=0.25, max_step_per_episode=64,
                           episodes_per_iter=ep_per_pev, MC_max_iteration=math.ceil(2000 / ep_per_pev),
                           MC_with_Q_initialization=True, MC_Q_error_threshold=0,
                           enable_sampling=True)
        mc_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_grid_agent_.train()
        return mc_grid_agent_

    @repeat(REPEAT_NUM)
    def mc_epsilon(epsilon: float):
        params_ = RLParams(name=f"MC epsilon={epsilon}",
                           gamma=0.95, epsilon=epsilon, lamda=0.25, max_step_per_episode=64,
                           episodes_per_iter=16, MC_max_iteration=150,
                           MC_with_Q_initialization=True, MC_Q_error_threshold=0,
                           enable_sampling=True)
        mc_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_grid_agent_.train()
        return mc_grid_agent_

    @repeat(REPEAT_NUM)
    def mc_lamda(lamda: float):
        params_ = RLParams(name=f"MC lambda={lamda}",
                           gamma=0.95, epsilon=0.1, lamda=lamda, max_step_per_episode=64,
                           episodes_per_iter=16, MC_max_iteration=150,
                           MC_with_Q_initialization=True, MC_Q_error_threshold=0,
                           enable_sampling=True)
        mc_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_grid_agent_.train()
        return mc_grid_agent_

    return {
        'dp': dp(),
        'mc': {
            'pev': [mc_pev(i) for i in [5, 20, 80]],
            'epsilon': [mc_epsilon(e) for e in [0.1, 0.3, 0.5]],
            'lamda': [mc_lamda(e) for e in [0.25, 0.5, 0.75]],
            'init': mc_with_init(),
            'without_init': mc_wo_init()
        }
    }

# ================ Code to calculate mean reward ================
def reward(agents):
    """
    :param agents: dict returned from `prepare_agents`
    :return: a dict of rewards
    """
    bga = BaseGridAgent(**common_problem_defs, rl_params=RLParams(name="Reward eval", max_step_per_episode=50, reward=(-1, 9)))

    def traverse_policy(policy):
        bga.policy = policy
        return bga.traverse_iterative()

    def traverse_policies(policies):
        return [traverse_policy(policies[..., i]) for i in range(policies.shape[-1])]

    def traverse_repeat(agents):
        all_reward = [traverse_policies(i.params.sampler.data['policy']) for i in agents]
        return np.mean(all_reward, axis=0), np.std(all_reward, axis=0), \
            np.max(all_reward, axis=0), np.min(all_reward, axis=0)

    def traverse_list(agents_group):
        return [traverse_repeat(i) for i in agents_group]

    return {
        'dp': traverse_policy(agents['dp'].policy),
        'mc': {
            'pev': traverse_list(agents['mc']['pev']),
            'epsilon': traverse_list(agents['mc']['epsilon']),
            'lamda': traverse_list(agents['mc']['lamda'])
        }
    }


# ================ utility function to save figure to fs ================
def save_fig(fig, name):
    """
    :param fig: matplotlib fig object
    :param name: filename
    """
    from inspect import getframeinfo, stack
    caller = getframeinfo(stack()[1][0])
    with open(OUTPUT_DIR + '/mapping.txt', 'a') as f:
        f.write(f"{os.path.basename(caller.filename)}\t\tL{caller.lineno}\t\t{name}\n")

    fig.savefig(OUTPUT_DIR + '/' + name, format=OUTPUT_FORMAT)
    # uncomment to view figure
    # fig.show()
    plt.close(fig)


# ================ all the plotting code ================
def plot(agents, rewards):
    """
    :param agents: dict returned from `prepare_agents`
    :param rewards: dict returned from `reward`
    """
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if os.path.exists(OUTPUT_DIR + '/mapping.txt'):
        os.unlink(OUTPUT_DIR + '/mapping.txt')
    fig_id = 0
    fig_size = (PLT_CONF.fig_size * PLT_CONF.figsize_scalar, PLT_CONF.fig_size * PLT_CONF.figsize_scalar)
    fig_size_sq = (PLT_CONF.fig_size_squre * PLT_CONF.figsize_scalar, PLT_CONF.fig_size_squre * PLT_CONF.figsize_scalar)

    # mc without init
    # initial policy and value
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi,)
    ax.set_position([0,0,1,1])
    plot_policy(agents['mc']['without_init'], 0, ax)
    #plt.tight_layout(pad=default_cfg.pad)
    save_fig(fig, str(fig_id)+'.'+OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
    plot_value(agents['mc']['without_init'], 0, ax)
    #plt.tight_layout(pad=default_cfg.pad)
    ax.set_position([0, 0, 1, 1])
    save_fig(fig, str(fig_id)+'.'+OUTPUT_FORMAT)
    fig_id += 1

    # initial route
    for i in range(4):
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
        plot_route(agents['mc']['without_init'], i, ax)
        #plt.tight_layout(pad=default_cfg.pad)
        ax.set_position([0, 0, 1, 1])
        save_fig(fig, str(fig_id)+'.'+OUTPUT_FORMAT)
        fig_id += 1

    # # save route as video, UNCOMMENT to run
    # fig, ax = plt.subplots()
    # ani = plot_route_animated(agents['mc']['without_init'], 1, fig, ax)
    # # Refer to https://github.com/matplotlib/matplotlib/issues/8794
    # ani.save(OUTPUT_DIR + '/03.mp4', writer="ffmpeg", extra_args=['-r', '25'])
    # plt.close(fig)

    # action stats
    for i in range(4):
        fig = plt.figure(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
        ax = fig.add_subplot(111, projection='3d')
        plt.tick_params(labelsize=default_cfg.tick_size)
        labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
        [label.set_fontname(default_cfg.tick_label_font) for label in labels]
        plot_statistics(agents['mc']['without_init'], 10*i, ax)
        plt.tight_layout()
        save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
        fig_id += 1

    # mc with init
    for i in range(1,5):
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
        plot_policy(agents['mc']['init'], i, ax)
        ax.set_position([0, 0, 1, 1])
        #plt.tight_layout(pad=default_cfg.pad)
        save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
        fig_id += 1

    for i in range(1,5):
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
        plot_value(agents['mc']['init'], i, ax)
        ax.set_position([0, 0, 1, 1])
        #plt.tight_layout(pad=default_cfg.pad)
        save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
        fig_id += 1

    # learned policy and value
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
    plot_policy(agents['mc']['init'], -1, ax)
    ax.set_position([0, 0, 1, 1])
    #plt.tight_layout(pad=default_cfg.pad)
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size_sq), dpi=PLT_CONF.dpi)
    plot_value(agents['mc']['init'], -1, ax)
    ax.set_position([0, 0, 1, 1])
    #plt.tight_layout(pad=default_cfg.pad)
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    # mc compare
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_rms(agents['mc']['pev'], agents['dp'], 'Ep/PEV=%d', [5, 20, 80], ax)
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_reward_pre(agents['mc']['pev'], rewards['mc']['pev'], rewards['dp'], 'Ep/PEV=%d', [5, 20, 80], ax)
    plt.ylim([-30.0, 5.0])
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_rms(agents['mc']['epsilon'], agents['dp'], R'$\epsilon$=%g', [0.1, 0.3, 0.5], ax)
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_reward_pre(agents['mc']['epsilon'], rewards['mc']['epsilon'], rewards['dp'], R'$\epsilon$=%g',
                    [0.1, 0.3, 0.5], ax)
    plt.ylim([-30.0, 5.0])
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_rms(agents['mc']['lamda'], agents['dp'], R'$\lambda$=%g', [0.25, 0.50, 0.75], ax)
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_reward_pre(agents['mc']['lamda'], rewards['mc']['lamda'], rewards['dp'], R'$\lambda$=%g',
                    [0.25, 0.50, 0.75], ax)
    plt.ylim([-30.0, 5.0])
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1


# ================ Plot figures ================
if __name__ == '__main__':
    AGENT_FILE = '%s/%s.pkl' % (DIR_NAME, AGENT_NAME)
    REWARD_FILE = '%s/%s.pkl' % (DIR_NAME, REWARD_NAME)

    import os

    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)

    if LOAD_PRETRAINED == 0:
        agents = prepare_agents()
        rewards = reward(agents)
        dump(AGENT_FILE, agents)
        dump(REWARD_FILE, rewards)
    elif LOAD_PRETRAINED == 1:
        agents = load(AGENT_FILE)
        rewards = reward(agents)
        dump(REWARD_FILE, rewards)
    else:
        agents = load(AGENT_FILE)
        rewards = load(REWARD_FILE)

    plot(agents, rewards)
    print('Plot finished!')
