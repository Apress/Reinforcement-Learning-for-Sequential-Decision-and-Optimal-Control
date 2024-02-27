import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import init

LOG2Pi = np.log(2 * np.pi)


class AgentConfig:
    cell_num = 128
    action_bound = 1.0

    # deterministic policy
    action_exploration_noise = 0.1


class StochasticPolicy(nn.Module, AgentConfig):
    """
    stochastic policy
    """
    def __init__(self, input_size, output_size):
        super(StochasticPolicy, self).__init__()
        self.out_size = output_size

        # initial parameters of actor
        log_std = np.zeros(output_size)
        self.log_sigma = torch.nn.Parameter(torch.Tensor(log_std))

        self.layers = nn.Sequential(
            nn.Linear(input_size, self.cell_num),
            nn.SELU(),  # elu
            nn.Linear(self.cell_num, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, output_size),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x):
        """
        Parameters
        ----------
        x: features, shape:[batch, feature dimension]

        Returns
        -------
        mu: mean of action
        sigma: covariance of action
        dist: Gaussian(mu, dist)
        """
        mu = self.layers(x) * self.action_bound
        sigma = torch.exp(self.log_sigma)
        dist = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(mu, sigma),
            reinterpreted_batch_ndims=1
        )
         # = Normal(mu, sigma)

        return mu, sigma, dist

    def _initialize_weights(self):
        """
        xavier initial of parameters
        """
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def choose_action(self, state):
        """
        choose_action according to current state
        Parameters
        ----------
        state:  np.array; shape (batch, N_S)

        Returns
        -------
        action: shape (batch, N_a)
        mu: action without stochastic, shape (batch, N_a)
        """
        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))

        mu, sigma, dist = self.forward(torch.Tensor(state))
        action = dist.rsample()
        action = action.detach().numpy()
        mu = mu.detach().numpy()

        # action clipping
        action = np.clip(action, -1., 1.)
        return action, mu

    def save_parameter(self, logdir, index=None):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        if index is not None:
            torch.save(self.state_dict(), os.path.join(logdir, "actor_{}.pth".format(index)))
        else:
            torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))


class DeterminePolicy(nn.Module, AgentConfig):
    """
    Deterministic policy
    """
    def __init__(self, input_size, output_size):
        super(DeterminePolicy, self).__init__()
        self.out_size = output_size

        # initial parameters of actor

        self.layers = nn.Sequential(
            nn.Linear(input_size, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, output_size),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Parameters
        ----------
        x: features, shape:[batch, feature dimension]

        Returns
        -------
        mu: mean of action
        """
        if len(x.shape) == 1:
            x = x.reshape((-1, x.shape[0]))

        mu = self.layers(x) * self.action_bound
        return mu

    def _initialize_weights(self):
        """
        xavier initial of parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def choose_action(self, state):
        """
        choose_action according to current state
        Parameters
        ----------
        state:  np.array; shape (batch, N_S)

        Returns
        -------
        action: shape (batch, N_a)
        mu: action without stochastic, shape (batch, N_a)
        """
        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))

        action = self.forward(torch.Tensor(state))
        mu = action.detach().numpy()
        
        # add exploration noise
        action = action + self.action_exploration_noise * torch.randn_like(action)
        action = action.detach().numpy()

        # action clipping
        action = np.clip(action, -1., 1.)
        return action, mu

    def save_parameter(self, logdir, index=None):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        if index is not None:
            torch.save(self.state_dict(), os.path.join(logdir, "actor_{}.pth".format(index)))
        else:
            torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))


class StateValue(nn.Module, AgentConfig):
    """
    state value function
    """
    def __init__(self, input_size, output_size):
        super(StateValue, self).__init__()

        self.out_size = output_size

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, output_size),
            nn.Identity()
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
        if len(x.shape) == 1:
            x = x.reshape((-1, x.shape[0]))

        y = self.layers(x)
        return y
    
    def _initialize_weights(self):
        """
        initial parameter using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)


class ActionValue(nn.Module, AgentConfig):
    """
    action value function
    """

    def __init__(self, input_size, output_size):
        super(ActionValue, self).__init__()

        self.out_size = output_size

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, self.cell_num),
            nn.SELU(),
            nn.Linear(self.cell_num, output_size),
            nn.Identity()
        )
        # initial optimizer
        self._initialize_weights()

    def forward(self, state, action):
        """
        Parameters
        ----------
        state : current state [batch, feature dimension]
        action: actions       [batch, action dimension]

        Returns
        -------
        value of current state_ction
        """
        if len(action.shape) == 1:
            action = action.reshape((-1, action.shape[0]))

        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))
        
        sa = torch.cat([state, action], dim=1)
        y = self.layers(sa)
        return y

    def _initialize_weights(self):
        """
        initial parameter using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)
