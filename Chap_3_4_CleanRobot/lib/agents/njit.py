from typing import Tuple
import numba
import numpy as np
from .. import is_valid_index

@numba.njit()
def state_to_position(grid_size: Tuple[int, int], state: int):
    return state // grid_size[1], state % grid_size[1]

@numba.njit()
def position_to_state(grid_size: Tuple[int, int], position: Tuple[int, int]):
    return position[0] * grid_size[1] + position[1]

@numba.njit()
def rand_choice_nb(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@numba.njit()
def move_rand(current_state, actions, action_delta, policy, agent_dynamics, grid_size, rewards, target_state):
    action = rand_choice_nb(actions, policy[current_state])
    return move(current_state, action, actions, action_delta, agent_dynamics, grid_size, rewards, target_state)

@numba.njit()
def move(current_state, action, actions, action_delta, agent_dynamics, grid_size, rewards, target_state):
    real_action = rand_choice_nb(actions, agent_dynamics[action])
    delta = action_delta[real_action]
    current_position = state_to_position(grid_size, current_state)
    next_position = (current_position[0] + delta[0], current_position[1] + delta[1])

    if not is_valid_index(next_position, grid_size):
        return current_state, action, rewards[0], False  # , real_action, True

    next_state = position_to_state(grid_size, next_position)
    done = next_state == target_state
    reward = rewards[1 if done else 0]
    return next_state, action, reward, done  # , real_action, False
