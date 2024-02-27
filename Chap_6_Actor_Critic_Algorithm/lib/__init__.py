from lib.agent import DeterminePolicy, StochasticPolicy, ActionValue, StateValue

# from lib.environment import EnvRandom

from lib.circle_road import GymLaneKeeping2D

from lib.utils import policy_evaluate

from lib.idplot import self_plot

from gym.envs.registration import register

register(id='Lanekeeping-v0', entry_point='lib.circle_road:GymLaneKeeping2D', max_episode_steps=1000)