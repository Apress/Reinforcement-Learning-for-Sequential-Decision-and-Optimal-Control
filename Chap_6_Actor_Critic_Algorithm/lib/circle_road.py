import warnings
import numpy as np
import torch
from gym.spaces import Box
from gym import Env
import math
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from matplotlib.backends.backend_agg import FigureCanvasAgg


RHO0 = 100 # road radius
DT = 0.05  # simpling time
EXPECTED_V = 15 # expected vehicle speed
MAX_DELTA = 20 * (math.pi / 180) # maximum front wheel angle (rad)
MAX_ACC = 1. # maximum acceleration (m/s^2)

def np_state(func):
    def wrapper(self, state, action):
        transform = False
        if isinstance(state, np.ndarray):
            assert isinstance(action, np.ndarray)
            transform = True

        if transform:
            state = torch.as_tensor(state[np.newaxis, :], dtype=torch.float32)
            action = torch.as_tensor(action[np.newaxis, :], dtype=torch.float32)

        state_next, info = func(self, state, action)
        if transform:
            state_next = state_next.numpy().squeeze(0)
            for k, v in info.items():
                info[k] = v[0].numpy()
        return state_next, info
    return wrapper


def np_reward(func):
    def wrapper(self, state, action):
        transform = False
        if isinstance(state, np.ndarray):
            assert isinstance(action, np.ndarray)
            transform = True

        if transform:
            state = torch.as_tensor(state[np.newaxis, :], dtype=torch.float32)
            action = torch.as_tensor(action[np.newaxis, :], dtype=torch.float32)

        reward, info = func(self, state, action)
        if transform:
            reward = reward[0].item()
            for k, v in info.items():
                info[k] = v[0].item()
        return reward, info
    return wrapper

def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


class GymLaneKeeping2D(Env):
    def __init__(self):
        self.dynamics = VehicleDynamic2D()
        
        # state-action dimension info
        state_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = Box(low=-state_high, high=state_high)
        action_high = np.array([1, 1], dtype=np.float32)
        self.action_space = Box(low=-action_high, high=action_high)

        # environment variable
        self.obs = None
        self.action = None
        self.theta = 0.

        # for plot
        self.rho0 = RHO0
        self.plot_flag = False
        self.model_info = None
        self.reward_info = None

        # done conditions
        self.d_lim = 5.
        self.phi_lim = math.pi / 3
        self.u_lim = 5.
        
        # action range
        self.max_delta = MAX_DELTA
        self.max_acc = MAX_ACC

    def reset(self, init_state = None):
        if init_state is None:
            rho = np.random.randn() * 0.7
            phi = np.random.randn() * np.pi / 18.
            v_x = np.random.randn() * 1.
            v_y = np.random.randn() * 0.1
            omega = np.random.randn() * 0.1

            rho = np.clip(rho, -2.1, 2.1)
            phi = np.clip(phi, -np.pi / 6 , np.pi / 6)
            v_x = np.clip(v_x, -3.0, 3.)
            v_y = np.clip(v_y, -0.3, 0.3)
            omega = np.clip(omega, -0.3, 0.3)
            self.obs = np.stack([rho, phi, v_x, v_y, omega])
        else:
            self.obs = init_state

        self.reset_F()

        self.theta = 0.
        return self.obs

    def step(self, action):
        obs = self.obs
        self.obs, self.model_info = self.dynamics.prediction(obs, action)
        self.theta = self.dynamics.np_theta_pred(self.theta, self.obs)
        r, self.reward_info = self.compute_reward(self.obs, action)
        d = self._done(self.obs)
        self.action = action
        return self.obs, r, d, {}

    def _done(self, sta):
        d = False
        if np.abs(sta[0]) > self.d_lim:
            d = True
        if np.abs(sta[1]) > self.phi_lim:
            d = True
        if np.abs(sta[2]) > self.u_lim:
            d = True
        return d

    @np_reward
    def compute_reward(self, state, action):
        action = torch.tensor([MAX_DELTA, MAX_ACC]) * action
        delta = action[:, 0]
        acc = action[:, 1]

        del_rho = state[:, 0]
        del_phi = state[:, 1]
        v_x = state[:, 2]
        v_y = state[:, 3]
        omega = state[:, 4]

        # punish_rho = del_rho * del_rho * 1.75 / (1.5 ** 2)
        # punish_phi = del_phi * del_phi * 0.25 / (0.35 ** 2)
        # punish_v_x = v_x * v_x * 1.5 / (2.0 ** 2)

        punish_rho = abs(del_rho) * 2.0 / 1.5  # L1 Loss
        punish_phi = del_phi * del_phi * 0.25 / (0.35 ** 2)
        punish_v_x = abs(v_x) * 1.25 / 2.0

        punish_v_y = v_y * v_y * 0.
        punish_omega = omega * omega * 0.
        punish_delta = (delta - 0.0) ** 2 * 1.0 / (self.max_delta ** 2)
        punish_acc = acc * acc * 0.5 / (self.max_acc ** 2)
        
        r = 10 - punish_rho - punish_phi - punish_v_x - punish_v_y - punish_omega - punish_delta - punish_acc
        if torch.abs(del_rho) > self.d_lim or \
           torch.abs(del_phi) > self.phi_lim or \
           torch.abs(v_x) > self.u_lim:
            r += -100

        reward_info = dict(punish_rho=punish_rho,
                           punish_phi=punish_phi,
                           punish_v_x=punish_v_x,
                           punish_v_y=punish_v_y,
                           punish_omega=punish_omega,
                           punish_delta=punish_delta,
                           punish_acc=punish_acc)
        return r, reward_info

    def reset_F(self):
        self.dynamics.reset_F()

    def reset_F_fix(self, F_noi=None):
        self.dynamics.reset_F_fix(F_noi)

    def render(self, mode='human'):
        LANE_WIDTH = 3.75

        def ploter(to_array=False):
            plt.cla()
            plt.ion()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            # ax = plt.axes([-35, -35, 35, 35])
            ax.axis("equal")

            # plot midline
            anglec = np.linspace(0, 2 * np.pi, 128)
            xc = self.rho0 * np.cos(anglec)
            yc = self.rho0 * np.sin(anglec)
            plt.plot(xc, yc, 'k--')
            # plot inner
            xinner = (self.rho0 - LANE_WIDTH / 2) * np.cos(anglec)
            yinner = (self.rho0 - LANE_WIDTH / 2) * np.sin(anglec)
            plt.plot(xinner, yinner, 'k')

            # plot outer
            xouter = (self.rho0 + LANE_WIDTH / 2) * np.cos(anglec)
            youter = (self.rho0 + LANE_WIDTH / 2) * np.sin(anglec)
            plt.plot(xouter, youter, 'k')

            def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
                # a = -a
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)
            # plot vehicle
            # print(type(self.obs))
            rho, phi, v_x, v_y, omega = self.obs
            theta = self.theta
            vehicle_a = (theta + phi + np.pi / 2) / (np.pi) * 180
            vehicle_x = (rho + self.rho0) * np.cos(theta)
            vehicle_y = (rho + self.rho0) * np.sin(theta)
            # print(radius0 + self.rho0)
            # print(type(vehicle_x), type(vehicle_y), type(vehicle_a))
            # print('---', type(radius0), type(self.rho0), type(theta))

            draw_rotate_rec(vehicle_x, vehicle_y, vehicle_a, 4.8, 2.0, 'r')

            text_x, text_y_start = -8., 25
            text_y = iter(range(0, 100, 4))
            # add text
            if self.reward_info:
                punish_rho = self.reward_info['punish_rho']
                punish_phi = self.reward_info['punish_phi']
                punish_v_x = self.reward_info['punish_v_x']
                punish_v_y = self.reward_info['punish_v_y']
                punish_omega = self.reward_info['punish_omega']
                punish_delta = self.reward_info['punish_delta']
                punish_acc = self.reward_info['punish_acc']

                plt.text(text_x, text_y_start - next(text_y), 'punish_rho: {:.2f}'.format(punish_rho))
                plt.text(text_x, text_y_start - next(text_y), 'punish_phi: {:.2f}'.format(punish_phi))
                plt.text(text_x, text_y_start - next(text_y), 'punish_v_x: {:.2f}'.format(punish_v_x))
                plt.text(text_x, text_y_start - next(text_y), 'punish_v_y: {:.2f}'.format(punish_v_y))
                plt.text(text_x, text_y_start - next(text_y), 'punish_omega: {:.2f}'.format(punish_omega))
                plt.text(text_x, text_y_start - next(text_y), 'punish_delta: {:.2f}'.format(punish_delta))
                plt.text(text_x, text_y_start - next(text_y), 'punish_acc: {:.2f}'.format(punish_acc))

            if self.obs is not None:
                del_rho, del_phi, v_x, v_y, omega = self.obs
                plt.text(text_x, text_y_start - next(text_y), 'del_rho: {:.2f}'.format(del_rho))
                plt.text(text_x, text_y_start - next(text_y), 'del_phi: {:.2f}'.format(del_phi))
                plt.text(text_x, text_y_start - next(text_y), 'v_x: {:.2f}'.format(v_x))
                plt.text(text_x, text_y_start - next(text_y), 'v_y: {:.2f}'.format(v_y))
                plt.text(text_x, text_y_start - next(text_y), 'omega: {:.2f}'.format(omega))

            if self.action is not None:
                delta, acc = self.action
                plt.text(text_x, text_y_start - next(text_y), 'delta: {:.2f}'.format(delta))
                plt.text(text_x, text_y_start - next(text_y), 'acc: {:.2f}'.format(acc))
            if not to_array:
                plt.show()
                plt.pause(0.001)
                return None
            else:
                canvas = FigureCanvasAgg(plt.gcf())
                canvas.draw()
                img = np.array(canvas.renderer.buffer_rgba())
                return img

        if mode == 'human':
            return ploter()
        elif mode == 'rgb_array':
            return ploter(to_array=True)


class VehicleDynamic2D():
    def __init__(self):
        self.param = dict(C_f=88000,
                          C_r=95000,
                          L_a=1.14,
                          L_b=1.40,
                          mass=1500,
                          I_zz=2420,
                          g=9.81,
                          mu=1.0)
        m = self.param['mass']
        g = self.param['g']
        L_a = self.param['L_a']
        L_b = self.param['L_b']
        L = self.param['L_a'] + self.param['L_b']
        F_zf = m * g * L_b / L
        F_zr = m * g * L_a / L

        fw1 = pow(self.param['C_f'], 2) / (3 * self.param['mu'] * F_zf)
        rw1 = pow(self.param['C_r'], 2) / (3 * self.param['mu'] * F_zr)

        fw2 = pow(self.param['C_f'], 3) / (27 * pow(self.param['mu'] * F_zf, 2))
        rw2 = pow(self.param['C_r'], 3) / (27 * pow(self.param['mu'] * F_zr, 2))
        self.param.update(F_zf=F_zf, F_zr=F_zr, fw1=fw1, fw2=fw2, rw1=rw1, rw2=rw2)

        # noise
        self.fw_low = 0
        self.fw_high = 100
        self.fw1_mu = 0
        self.fw1_sig = 50
        self.fw2_mu = 0
        self.fw2_sig = 50 * 4
        self.a_x_sig = 0.05

        # action range
        self.max_delta = MAX_DELTA
        self.max_acc = MAX_ACC

    def fiala(self, v_x, v_y, omega, delta, acc):
        L_a = self.param['L_a']
        L_b = self.param['L_b']
        C_f = self.param['C_f']
        C_r = self.param['C_r']

        fw1 = self.param['fw1']
        rw1 = self.param['rw1']
        fw2 = self.param['fw2']
        rw2 = self.param['rw2']

        F_zf = self.param['F_zf']
        F_zr = self.param['F_zr']

        mu = self.param['mu']
        alpha_f = torch.atan((v_y + L_a * omega) / v_x) - delta
        alpha_r = torch.atan((v_y - L_b * omega) / v_x)

        F_yf = - C_f * torch.tan(alpha_f) \
               + fw1 * torch.mul(torch.tan(alpha_f), torch.abs(torch.tan(alpha_f))) \
               - fw2 * torch.pow(torch.tan(alpha_f), 3)
        F_yr = - C_r * torch.tan(alpha_r) \
               + rw1 * torch.mul(torch.tan(alpha_r), torch.abs(torch.tan(alpha_r))) \
               - rw2 * torch.pow(torch.tan(alpha_r), 3)

        F_yf_max = torch.tensor(F_zf * mu, dtype=torch.float32)
        F_yr_max = torch.tensor(F_zr * mu, dtype=torch.float32)
        F_yf[F_yf > F_yf_max] = F_yf_max
        F_yf[F_yf < -F_yf_max] = -F_yf_max

        F_yr[F_yr > F_yr_max] = F_yr_max
        F_yr[F_yr < -F_yr_max] = -F_yr_max
        return alpha_f, alpha_r, F_yf, F_yr

    def f(self, state, action):
        # [rho, phi, v_x, v_y, omega]
        rho = state[:, 0]
        phi = state[:, 1]
        v_x = state[:, 2]
        v_y = state[:, 3]
        omega = state[:, 4]

        delta = action[:, 0]
        acc = action[:, 1]

        m = self.param['mass']
        L_a = self.param['L_a']
        L_b = self.param['L_b']
        I_zz = self.param['I_zz']
        alpha_f, alpha_r, F_yf, F_yr = self.fiala(v_x, v_y, omega, delta, acc)
        
        # add noise
        F_yf += np.random.randn(*F_yf.shape).astype(np.float32) * self.fw1_sig
        F_yr += np.random.randn(*F_yr.shape).astype(np.float32) * self.fw2_sig
        acc += np.random.randn(*acc.shape).astype(np.float32) * self.a_x_sig

        rho_dot = - v_x * torch.sin(phi) - v_y * torch.cos(phi)
        phi_dot = omega - (v_x * torch.cos(phi) - v_y * torch.sin(phi)) / rho
        v_x_dot = acc + v_y * omega
        v_y_dot = (F_yf * torch.cos(delta) + F_yr) / m - v_x * omega
        omega = (L_a * F_yf * torch.cos(delta) - L_b * F_yr) / I_zz

        # concate them
        f_xu = torch.stack((rho_dot, phi_dot, v_x_dot, v_y_dot, omega), dim=1)
        state_new = state + f_xu * DT
        return state_new, {}

    @np_state
    def prediction(self, state, action):
        rho_old = state[:, 0] + RHO0
        phi_old = state[:, 1]
        v_x_old = state[:, 2] + EXPECTED_V
        v_y_old = state[:, 3]
        omega_old = state[:, 4]

        state_new = torch.ones_like(state)

        state_abs = torch.stack([rho_old, phi_old, v_x_old, v_y_old, omega_old], dim=1)

        action = torch.tensor([MAX_DELTA, MAX_ACC]) * action
        state_abs_new, model_info = self.f(state_abs, action)

        state_new[:, 0] = state_abs_new[:, 0] - RHO0
        state_new[:, 1] = state_abs_new[:, 1]
        state_new[:, 2] = state_abs_new[:, 2] - EXPECTED_V
        state_new[:, 3] = state_abs_new[:, 3]
        state_new[:, 4] = state_abs_new[:, 4]
        return state_new, model_info

    def np_theta_pred(self, theta, state):
        rho_old = state[0] + RHO0
        phi_old = state[1]
        v_x_old = state[2] + EXPECTED_V
        v_y_old = state[3]
        omega_old = state[4]

        theta_dot = (v_x_old * np.cos(phi_old) - v_y_old * np.sin(phi_old)) / rho_old
        return theta + theta_dot * DT

    def reset_F(self):
        self.fw1_mu = np.random.uniform(low=self.fw_low, high=self.fw_high)
        self.fw2_mu = self.fw1_mu

    def reset_F_fix(self, F_noi=None):
        if F_noi is None:
            F_noi = self.fw_high
        self.fw1_mu = F_noi
        self.fw2_mu = F_noi

def test2():
    env = GymLaneKeeping2D()
    env.reset()

    for _ in range(100):
        a = env.action_space.sample()
        a = np.array([0, 0])
        s, _, _, _ = env.step(a)
        print(s)

if __name__ == "__main__":
    # t_gym()
    test2()