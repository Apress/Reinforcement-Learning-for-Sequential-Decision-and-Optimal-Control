"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Yuxuan Jiang

Description: Chapter 3 & 4:  Plot figures for DP, MC, Sarsa, and Q-learning algorithms

"""

from common_plot import *


# ================ Select Option ================
LOAD_PRETRAINED = 2
# 0 - calculate both;
# 1 - load agents, calculate returns;
# 2 - load both agents and returns.
REPEAT_NUM = 10
OUTPUT_DIR = OUTPUT_ROOT_DIR + "/MC_Sarsa_QL"
DIR_NAME = DIR_ROOT_NAME + "/MC_Sarsa_QL"

# ================ Prepare all agents for plotting ================
def prepare_agents():
    def dp():
        params_ = RLParams(name="DP", gamma=0.95, DP_max_iteration=100,
                           DP_error_threshold=1e-8)
        dp_grid_agent_ = DPGridAgent(**common_problem_defs, rl_params=params_)
        dp_grid_agent_.train()
        return dp_grid_agent_

    @repeat(REPEAT_NUM)
    def compare_mc():
        params_ = RLParams(name="COMPARE MC", gamma=0.95, epsilon=0.1, lamda=0.50,
                           max_step_per_episode=64, episodes_per_iter=16, MC_max_iteration=64, 
                           MC_with_Q_initialization=True, MC_Q_error_threshold=0,
                           enable_sampling=True)
        mc_grid_agent_ = MCGridAgent(**common_problem_defs, rl_params=params_)
        mc_grid_agent_.train()
        return mc_grid_agent_

    @repeat(REPEAT_NUM)
    def compare_sarsa():
        params_ = RLParams(name="COMPARE Sarsa", alpha=0.1, gamma=0.95, max_episodes=1000,
                           max_step_per_episode=500000, Sarsa_PEV_steps=1, Sarsa_TD_steps=1, enable_sampling=True)
        sarsa_grid_agent_ = SarsaGridAgent(**common_problem_defs, rl_params=params_)
        sarsa_grid_agent_.train()
        return sarsa_grid_agent_

    @repeat(REPEAT_NUM)
    def compare_ql():
        params_ = RLParams(name="COMPARE QL", alpha=0.1, gamma=0.95, max_episodes=1000,
                           max_step_per_episode=500000, QL_init_coefficient=1, enable_sampling=True)
        ql_grid_agent_ = QLGridAgent(**common_problem_defs, rl_params=params_)
        ql_grid_agent_.train()
        return ql_grid_agent_

    return {
        'dp': dp(),
        'COMPARE': {
            'mc': compare_mc(),
            'sarsa': compare_sarsa(),
            'ql': compare_ql()
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
        'COMPARE': {
            'mc': traverse_repeat(agents['COMPARE']['mc']),
            'sarsa': traverse_repeat(agents['COMPARE']['sarsa']),
            'ql': traverse_repeat(agents['COMPARE']['ql'])
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

    # compare mc, sarsa, ql
    agents_to_compare = [agents['COMPARE']['mc'], agents['COMPARE']['sarsa'], agents['COMPARE']['ql']]
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_rms(agents_to_compare, agents['dp'], '%s', ['MC', 'Sarsa', 'Q-Learn'], ax)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.legend(loc='best', ncol=1, prop=default_cfg.legend_font)
    plt.tight_layout()
    save_fig(fig, str(fig_id) + '.' + OUTPUT_FORMAT)
    fig_id += 1

    rewards_to_compare = [rewards['COMPARE']['mc'], rewards['COMPARE']['sarsa'], rewards['COMPARE']['ql']]
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PLT_CONF.dpi)
    plot_reward_pre(agents_to_compare, rewards_to_compare, rewards['dp'], '%s', ['MC', 'Sarsa', 'Q-Learn'], ax)
    plt.ylim([-30.0, 5.0])
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
