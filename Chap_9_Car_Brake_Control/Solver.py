"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Yuhang Zhang

Description: Chapter 9:  OCP example for emergency braking control
             Closed loop optimization (i.e., model predictive control)

"""
from casadi import *
import math


class Solver():
    """
    NLP solver for nonlinear model predictive control with Casadi.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.U_UPPER = 0
        self._sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.zero = [0., 0.]
        self.x_last = 0

        super(Solver, self).__init__()

    def mpcSolver(self, x_init):
        """
        Solver of nonlinear MPC

        Parameters
        ----------
        x_init: list
            input state for MPC.
        predict_steps: int
            steps of predict horizon.
        CBF: bool
            whether using control barrier function constraints
        cbf_para:
            float: lambda

        Returns
        ----------
        state: np.array     shape: [predict_steps+1, state_dimension]
            state trajectory of MPC in the whole predict horizon.
        control: np.array   shape: [predict_steps, control_dimension]
            control signal of MPC in the whole predict horizon.
        """

        x = SX.sym('x', self.cfg["DYNAMICS_DIM"])
        u = SX.sym('u', self.cfg["ACTION_DIM"])

        # lateral ACC model
        self.f_l = vertcat(
            x[0] + self.cfg["Ts"] * (- x[1]),
            x[1] + self.cfg["Ts"] * ( u[0] )
        )

        # Create solver instance

        self.F = Function("F", [x, u], [self.f_l])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.cfg["DYNAMICS_DIM"])
        w += [Xk]
        lbw += x_init
        ubw += x_init
        for k in range(1, self.cfg["Np"] + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.cfg["ACTION_DIM"])
            w += [Uk]
            lbw += [self.cfg["U_LOWER"]]
            ubw += [self.U_UPPER]

            Fk = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.cfg["DYNAMICS_DIM"])


            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += self.zero
            ubg += self.zero
            w += [Xk]
            if self.cfg["CBF"]:
                if k == 2:
                    constraint = (1-self.cfg["cbf_para"]) * (1-self.cfg["cbf_para"]) * (x_init[0] - self.cfg["d_safe"])
                    lbw += [constraint , 0]
                else:
                    lbw += [-20,0]
            else:
                lbw += [self.cfg["d_safe"], 0]
            ubw += [inf, 50]


            # Cost function
            F_cost = Function('F_cost', [x, u], [1 * (u[0]) ** 2])
            # F_cost = Function('F_cost', [x, u], [1 * (x[1] - x_init[1]) ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])


        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r)
        # print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([self.cfg["Np"], self.cfg["DYNAMICS_DIM"]])
        control = np.zeros([self.cfg["Np"] , self.cfg["ACTION_DIM"]])
        nt = self.cfg["DYNAMICS_DIM"] + self.cfg["ACTION_DIM"]  # total variable per step

        # save trajectories
        for i in range(self.cfg["Np"]):
            state[i] = state_all[nt * i: nt * i + nt - 1].reshape(-1)
            control[i] = state_all[nt * i + nt - 1]
        return state, control
