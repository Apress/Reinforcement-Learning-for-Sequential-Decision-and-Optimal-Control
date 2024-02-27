import matplotlib.pyplot as plt
import numpy as np

PI = 3.1415926


class StateConfig:
    a = 1.14
    L = 2.54
    b = L - a
    m = 1500.
    I_zz = 2420.0
    B = 14.0
    C = 1.43
    g = 9.81
    mu = 1.0
    F_z1 = m * g * b / L
    F_z2 = m * g * a / L

    # reset parameters
    rho_epect = 100.0
    rho_var = 1.6
    rho_range = 1.7
    psi_var = np.pi / 9
    beta_var = 0.01
    omega_var = 0.01
    dt = 0.05
    u = 15

    # reset_fix
    rho_init_fix = 0 + 100
    theta_init_fix = 0.0
    psi_init_fix = 0.0
    beta_init_fix = 0.0
    omega_init_fix = 0.0

    #  env random
    fw_low = 0
    fw_high = 100

    fw1_mu = 0
    fw1_sig = 50
    fw2_mu = 0
    fw2_sig = 50 * 4
    fw1_noise_old = 0
    fw2_noise_old = 0

    u0 = 10
    u_sin_co = 2.0
    u_sin_f = 1 / 5
    u_theta = 0


class EnvRandom(StateConfig):
    def __init__(self):
        self._state = np.zeros((1, 5))
        self.clock = 0
        raise NotImplementedError('Do not use this class')

    def _state_function(self, state2d, control):
        """
        non-linear model of the vehicle
        Parameters
        ----------
        state2d : np.array
            shape: [batch, 2], state of the state function
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        gradient of the states
        """

        if len(control.shape) == 1:
            control = control.reshape(1, -1)

        # input state
        beta = state2d[:, 0]
        omega_r = state2d[:, 1]

        # control
        delta = control[:, 0]

        alpha_1 = -delta + np.arctan(beta + self.a * omega_r / self.u)
        alpha_2 = np.arctan(beta - self.b * omega_r / self.u)

        # low-pass filter
        F_w1_noise = 2 * self.dt * (np.random.randn(*alpha_1.shape) * self.fw1_sig - self.fw1_noise_old) + self.fw1_noise_old
        F_w2_noise = 2 * self.dt * (np.random.randn(*alpha_2.shape) * self.fw2_sig - self.fw2_noise_old) + self.fw2_noise_old
        self.fw1_noise_old = F_w1_noise
        self.fw1_noise_old = F_w2_noise

        F_w1 = self.fw1_mu + F_w1_noise
        F_w2 = self.fw2_mu + F_w2_noise

        F_y1 = -self.mu * self.F_z1 * np.sin(self.C * np.arctan(self.B * alpha_1)) + F_w1
        F_y2 = -self.mu * self.F_z2 * np.sin(self.C * np.arctan(self.B * alpha_2)) + F_w2

        deri_beta = (np.multiply(F_y1, np.cos(delta)) + F_y2) / (self.m * self.u) - omega_r
        deri_omega_r = (np.multiply(self.a * F_y1, np.cos(delta)) - self.b * F_y2) / self.I_zz

        deri_state = np.concatenate((deri_beta[np.newaxis, :], deri_omega_r[np.newaxis, :]), 0)

        return deri_state.transpose(), F_y1, F_y2, alpha_1, alpha_2

    def _sf_with_axis_transform(self, control):
        """
        state function with the axis transform, the true model of RL problem
        Parameters
        ----------
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        state_dot ： np.array
            shape: [batch, 4], the gradient of the state
        """
        # state [\rho, \theta, \psi, \beta, \omega]
        assert len(self._state.shape) == 2


        rho = self._state[:, 0]
        theta = self._state[:, 1]
        psi = self._state[:, 2]
        beta = self._state[:, 3]
        omega = self._state[:, 4]

        rho_dot = -self.u * np.sin(psi) - self.u * np.tan(beta) * np.cos(psi)
        theta_dot = (self.u * np.cos(psi) - self.u * np.tan(beta) * np.sin(psi)) / rho
        psi_dot = omega - theta_dot
        state2d = self._state[:, 3:].reshape(1, -1)
        state2d_dot, _, _, alpha_1, alpha_2 = self._state_function(state2d, control)
        state_dot = np.concatenate([rho_dot[:, np.newaxis],
                                    theta_dot[:, np.newaxis],
                                    psi_dot[:, np.newaxis],
                                    state2d_dot], axis=1)

        return state_dot, alpha_1, alpha_2

    def reset(self):
        """
        reset the environment
        Returns
        -------
        s: np.array
            shape: [batch, 4], the initial state of the environment
        """
        self._state[0, 0] = self.rho_epect + np.random.randn(1)[0] * 0.8 * self.rho_var / 3  # rho
        self._state[0, 1] = np.random.uniform(0, 2 * PI, 1)[0]  # theta
        self._state[0, 2] = np.random.randn(1)[0] * self.psi_var * 0.3  # psi
        self._state[0, 3] = np.random.randn() * self.beta_var  # beta
        self._state[0, 4] = np.random.randn() * self.omega_var  # omega
        self.reset_u()
        self.reset_F()
        s = self._state[0, [0, 2, 3, 4]]

        s[0] -= self.rho_epect
        return s

    def reset_eval(self):
        """
        reset the environment
        Returns
        -------
        s: np.array
            shape: [batch, 4], the initial state of the environment
        """
        self._state[0, 0] = self.rho_epect + np.random.randn(1)[0] * 0.8 * self.rho_var / 3  # rho
        self._state[0, 1] = np.random.uniform(0, 2 * PI, 1)[0]  # theta
        self._state[0, 2] = np.random.randn(1)[0] * self.psi_var * 0.3   # psi
        self._state[0, 3] = np.random.randn() * self.beta_var # beta
        self._state[0, 4] = np.random.randn() * self.omega_var  # omega
        self.reset_u()
        self.reset_F()
        s = self._state[0, [0, 2, 3, 4]]

        s[0] -= self.rho_epect
        return s
    
    def reset_define(self, s):
        self._state[0, 0] = s[0] + self.rho_epect # rho
        self._state[0, 1] = np.random.uniform(0, 2 * PI, 1)[0]  # theta
        self._state[0, 2] = s[1]   # psi
        self._state[0, 3] = s[2]  # beta
        self._state[0, 4] = s[3]  # omega
        self.reset_u()
        self.reset_F()
        s = self._state[0, [0, 2, 3, 4]]

        s[0] -= self.rho_epect
        return s

    def step(self, action):
        """
        The environment will transform to the next state
        Parameters
        ----------
        action : np.array
            shape: [batch, 1]

        Returns
        -------

        """
        self.chang_u()
        state_dot, alpha_1, alpha_2 = self._sf_with_axis_transform(action)
        self._state += state_dot * self.dt

        # reward
        r = self._reward(action)

        # done
        done = False
        if np.abs(self._state[0, 0] - self.rho_epect) > self.rho_range:
            done = True

        # \theta is not a state
        s = self._state[0, [0, 2, 3, 4]]

        # \theta is useful when plotting the trajectory
        mask = self._state[0, 1]
        s[0] -= self.rho_epect
        return s, r, done, mask, (alpha_1, alpha_2)

    def _reward(self, action):
        """
        Output the reward of the step
        Parameters
        ----------
        action : np.array
            shape: [batch, 1]

        Returns
        -------
        reward
        """

        r = 10
        r -= 7 * np.power(self._state[0, 0] - self.rho_epect, 2)
        r -= 80 * np.power(action[0], 2)
        if np.abs(self._state[0, 0] - self.rho_epect) > self.rho_range:
            r = -500
        return r

    def reset_fix(self, F_noi=100):
        """
        reset the environment
        Returns
        -------
        s: np.array
            shape: [batch, 4], the initial state of the environment
        """
        self.reset_u_fix()
        self.reset_F_fix(F_noi)
        self._state[0, 0] = self.rho_init_fix  # rho
        self._state[0, 1] = self.theta_init_fix  # theta
        self._state[0, 2] = self.psi_init_fix   # psi
        self._state[0, 3] = self.beta_init_fix  # beta
        self._state[0, 4] = self.omega_init_fix  # omega

        s = self._state[0, [0, 2, 3, 4]]
        s[0] -= self.rho_epect
        return s

    def chang_u(self):
        self.u = self.u0 + self.u_sin_co * np.sin(2 * 3.1415926 * self.u_sin_f * self.clock + self.u_theta)
        self.clock += 1.0 * self.dt

    def reset_u(self):
        self.clock = 0.0
        self.u_theta = np.random.uniform(low=0, high=2 * 3.1415926)
        self.chang_u()


    def reset_u_fix(self):
        self.clock = 0.0
        self.u_theta = 0.0
        self.chang_u()

    def reset_F(self):
        self.fw1_mu = np.random.uniform(low=self.fw_low, high=self.fw_high)
        self.fw2_mu = self.fw1_mu

        self.fw1_noise_old = 0
        self.fw2_noise_old = 0

    def reset_F_fix(self, F_noi):
        self.fw1_mu = F_noi
        self.fw2_mu = F_noi

        self.fw1_noise_old = 0
        self.fw2_noise_old = 0

if __name__ == "__main__":
    env = EnvRandom()
    s = env.reset()
    print(s.shape)
    a = np.array([0])
    s, _, _, _, _ = env.step(a)

    print(s)