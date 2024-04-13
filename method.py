
# Method specification library for controlling uncertain dynamical system with probabilistic forecasting
# Copyright (c) 2021, Kiet Tuan Hoang/ Christian Ankerstjerne Thilker
# Last edited: 25.09.2023

""" References (for further information, check the references therein) """
# Model predictive control - J. Rawlings, D.Q. Mayne, and M. Diehl.Model Predictive Control: Theory, Computation,
#                            and Design.  Jan. 2017
# Advanced forecasting - C.A. Thilker, H. Madsen, J.B. Jørgensen, Advanced forecasting and disturbance modelling
#                        for model predictive control of smart energy systems, Applied Energy, 2021, ISSN 0306-2619,
#                        https://doi.org/10.1016/j.apenergy.2021.116889.
# Stochastic model predictive control - A. Mesbah, Stochastic Model Predictive Control: An Overview and Perspectives
#                                       for Future Research, in IEEE Control Systems Magazine, 2016,
#                                       doi: 10.1109/MCS.2016.2602087


''' Abbreviations '''
# General model predictive control functions are abbreviated with _mpc
# Economic model predictive control functions are abbreviated with _empc
# Stochastic model predictive control functions are abbreviated with _sempc
# Stochastic differential equations are abbreviated with _sode
# Kalman filter equations and functions are abbreviated with _kf
# Gas turbine generator is abbreviated with gtg
# Wind turbine generator is abbreviated with wtg
# Battery is abbreviated with bat

''' General notation '''
# 1. Class methods or symbolic functions are denoted with just their name: name
# 2. Class variables are denoted with an underscore: _name
# 3. Functions with _c before the names are related to the controller
# 4. Functions with _d before the names are related to the forecast method
# 6. Variables with _L after the names are in the Lamperti domain
# 7. Variables with _c after the names are related to the controller
# 8. Variables with _d after the names are related to the disturbance forecasting method
# 9. Variables with _cc after the names are related to stochastic chance constraints
# 10. c_horizon_c means certainty_horizon for the controller (exception)

"""
Dependencies information (libraries)
* numpy  : a general library for numerical operations
* casadi : a general library for symbolic operations
* scipy  : a general library for numerical integration and formulas
* alive_progress : a general library for creating a progress bar
"""

''' Imports '''
import numpy as np
import casadi as ca
import scipy.stats as ss
from scipy.integrate import odeint
from alive_progress import alive_bar

"""
Dependencies information (modules)
config : a library for utility functions and configurations

* system_bat : a function, used to compute the state of charge, the voltage and the power output of a battery
* system_wtg : a function, used to compute the power output of a wind turbine generator
* system_gtg : a function, used to compute the power output of a gas turbine generator
"""

''' Module imports '''
from system import system_bat
from system import system_wtg
from system import system_gtg
from system import logit


''' Stochastic differential eqauaitons in the Lamperti domain '''
def d_dfdx_L(x, wp, dwp, pars):
    theta_v_d, theta_R_d, theta_P_d, theta_Q_d, rho_d, mu_d, _, _, \
    xi1_d, xi2_d, xi3_d, sigmaZ, sigma_R_d, sigma_P_d, sigma_Q_d = pars;
    v = x[0]; r = x[1]; P_wtg = x[2]; q = x[3];

    _A = np.array([[(2 * (rho_d * dwp + r) * np.exp(-v ** 2 / 4) * v ** 2 - 2 * theta_v_d * v ** 2) / (2 * v ** 2) - 4 * (
                rho_d * dwp + r) * (1 - np.exp(-v ** 2 / 4) + theta_v_d * (wp * mu_d - v ** 2) - sigmaZ ** 2) / (2 * v ** 2),
                    (4 - 4 * np.exp(-v ** 2 / 4)) / (2 * v), 0, 0],
                   [0, -theta_R_d, 0, 0],
                   [theta_P_d * xi3_d * xi1_d * v * np.exp(-xi1_d * (v ** 2 / 4 + q - xi2_d)) / (
                               (1 + np.exp(-xi1_d * (v ** 2 / 4 + q - xi2_d))) ** 2 * np.cos(P_wtg)), 0,
                    -theta_P_d + 2 * theta_P_d * (
                                xi3_d / (1 + np.exp(-xi1_d * (v ** 2 / 4 + q - xi2_d))) - 0.5 - 0.5 * np.sin(P_wtg)) * np.sin(
                        P_wtg) / np.cos(P_wtg) ** 2, 2 * theta_P_d * xi3_d * xi1_d * np.exp(-xi1_d * (v ** 2 / 4 + q - xi2_d)) / (
                                (1 + np.exp(-xi1_d * (v ** 2 / 4 + q - xi2_d))) ** 2 * np.cos(P_wtg))],
                   [0, 0, 0, -theta_Q_d]]);
    return _A

def d_f_L(x, t, wp, dwp, pars):
    theta_v_d, theta_R_d, theta_P_d, theta_Q_d, rho_d, mu_d, _, _, xi1_d, xi2_d, xi3_d, sigmaZ, sigma_R_d, sigma_P_d, sigma_Q_d = pars
    v = x[0];
    r = x[1];
    P_wtg = x[2];
    q = x[3];
    P = x[4:].reshape(4, 4);

    A = d_dfdx_L(x, wp, dwp, pars)
    Q = np.array([[sigmaZ, 0, 0, 0],
                   [0, sigma_R_d, 0, 0],
                   [0, 0, sigma_P_d, 0],
                   [0, 0, 0, sigma_Q_d]])

    dv = (4 * (rho_d * dwp + r) * (1 - np.exp(-v ** 2 / 4)) + theta_v_d * (4 * wp * mu_d - v ** 2) - sigmaZ ** 2) / (2 * v)
    dr = -theta_R_d * r
    dvp = (2 * theta_P_d * (xi3_d / (1 + np.exp(-xi1_d * (v ** 2 / 4 - xi2_d + q))) - 0.5 * (1 + np.sin(P_wtg)))) / np.cos(P_wtg)
    dq = -theta_Q_d * q
    dP = A @ P + P @ A.T + Q @ Q.T

    der = np.concatenate(([dv, dr, dvp, dq],
                          dP.reshape(16, )))

    return der


''' System class '''
class method_class:
    def __init__(self, system_info, system_var,c_f, c_h, c_L, d_dhdx, d_h):
        """ Initialize dynamic controller
            # Arguments:
                system_info : a dictionary with all of the relevant system information
                              such as system dimensions, sampling time, process and measurment
                              disturbance standard deviation, initial values ...
                system_var  : a dictionary with all of the symbolic variables
                c_f         : state transition function - dxdt = f(x, u, p) for the controller
                c_h         : symbolic measurement function - y = h(x, u, p) for the controller
                c_L         : combined symbolic cost function and state transition function for the controller
                d_dhdx      : derivatve of h, d_dhdx - dhdx(y) for the disturbance forecast method
                d_h         : symbolic measurement function - y = h(x, u, p) for the disturbance forecast method
        """

        """ Initial initialization"""
        # Save system information
        self._system_info, self._system_var = system_info, system_var;
        self.c_L, self.c_f, self.c_h = c_L, c_f, c_h;
        self.d_dhdx, self.d_h = d_dhdx, d_h;

        """ Get the nominal and stochastic controllers """
        self.c_integrator = self.get_c_integrator_mpc(c_L=self.c_L); # Get the symbolic discrete integrator
        self._c_IPOPT_options = self.get_c_IPOPT_solver_options(solver=system_info['IPOPT_solver']); # Get IPOPT solver options

        self.c_empc, self._c_empc_info = self.get_c_empc_controller(); # Get the nominal economic model predictive controller
        self.c_sempc, self._c_sempc_info = self.get_c_sempc_controller(); # Get the stochastic economic model predictive controller

    def get_c_integrator_mpc(self,c_L):
        """ Create discrete integrator for the model predictive controller
            # Input:
                c_L          : combined symbolic cost function and state transition function
            # Outputs:
                c_integrator : an integrator for simulating the controller system dynamics and cost function
        """
        # Generate the differential values
        [k1, k1_cost] = c_L(self._system_var['x_c'], self._system_var['u_c'], self._system_var['s_c']);

        # Assemble them into the integrated system states with Eulers method
        X = self._system_var['x_c'] + self._system_info['dt'] * k1;
        Q = self._system_info['dt'] * k1_cost;

        # Create symbolic function for faster evaluations
        c_integrator = ca.Function('c_integrator', [self._system_var['x_c'], self._system_var['u_c'],
                                                    self._system_var['s_c']], [X, Q]);
        return c_integrator

    def get_c_IPOPT_solver_options(self,max_iter =10000, tolerance=1e-8, solver='mumps',print_level = 0):
        """ Get IPOPT options values for the controller
            # Arguments (Optional):
                max_iter      : maximum amount of iterations
                tolerance     : desired relative convergence tolerance
                solver        : linear solver employed inside of IPOPT for Newton ('mumps','ma27')
                print_level   : output verbosity complexity, higher the more complex [0,12]
            # Outputs:
                IPOPT_options : a dictionary containing specific IPOPT options, and casadi options
        """
        # Initialize dictionary
        c_IPOPT_options = {};

        # Basic IPOPT configs
        c_IPOPT_options["ipopt.max_iter"] = max_iter; # Maximum number of iterations
        c_IPOPT_options["ipopt.tol"] = tolerance; # Desired convergence tolerance (relative)
        c_IPOPT_options['ipopt.linear_solver'] = solver; # Solver used for step computations
        c_IPOPT_options['ipopt.print_level'] = print_level; # Output verbosity level
        print('IPOPT is being used with the linear solver: '+solver);

        # Expand symbolic framework from MX -> SX for speed (in case the author forgot to use SX..)
        c_IPOPT_options["expand"] = True;

        # Add warms starting option for IPOPT
        c_IPOPT_options["ipopt.warm_start_init_point"] = "yes";

        # Suppress native casadi printing
        c_IPOPT_options["print_time"] = 0;

        return c_IPOPT_options

    def get_c_empc_controller(self):
        """ Create discrete economic model predictive controller based on multiple shooting
            # Outputs:
                c_empc      : a program instance, computing optimal input for the economic
                              model predictive controller
                c_empc_info : a dictionary containing the economic model predictive control constraints
                              information and initial values
        """

        """ Initialize the program """
        # Initialize the data structures for constraints, optimization variables and cost function
        # w = decision variables, w0 = initial values for w, lbw = lower bounds for w, ubw = upper bounds for w
        # J = cumulative cost, g = inequality constraints, lbg = lower bounds for g, ubg = upper bounds for g
        w, w0, lbw, ubw, J, g, lbg, ubg = [], [], [], [], 0, [], [], [];

        # "Lift" initial conditions
        Xk = ca.SX.sym('X0_c', self._system_info['nX_c']);
        Sk = ca.SX.sym('S0_c', self._system_info['nS_c']);

        # Initial constraints
        w += [Xk, Sk]; w0 += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
        lbw += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
        ubw += (self._system_info['x0_c'] + [np.inf] * self._system_info['nS_c']);

        """ Generate the whole program """
        print('EMPC is being initialized, using multiple shooting scheme for integrating the dynamics')
        # Compute the multiple shooting branch
        J1, w1, g1, w01, lbg1, ubg1, lbw1, ubw1, R1, P1 = \
            self.get_c_empc_multiple_shooting_branch(Xk, Sk, self._system_info['discretization_c']);

        # Update current program
        J, w, g, w0 = J + J1, w + w1, g + g1, w0 + w01;
        lbw, ubw, lbg, ubg = lbw + lbw1, ubw + ubw1, lbg + lbg1, ubg + ubg1;

        """ Create the economic model predictive controller instance """
        # Create a program solver instance
        c_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': ca.vertcat(*P1, *R1)};
        # Create the economic model predictive controller
        c_empc = ca.nlpsol('c_solver_empc', 'ipopt', c_prob, self._c_IPOPT_options);
        # Save economic model predictive control constraints information and initial values
        c_empc_info = {'w0': w0, 'lbw': lbw, 'ubw': ubw, 'lbg': lbg, 'ubg': ubg};

        return c_empc, c_empc_info

    def get_c_empc_multiple_shooting_branch(self, Xk, Sk, N):
        """ Compute a whole multiple shooting branch for the economic model predictive controller
            # Arguments:
                Xk   : current state node [nX]
                Sk   : current slack state node [nS]
                N    : # prediction horizon for this shooting branch [1]
            # Outputs:
                J    : cumulative cost associated to this branch [1]
                w    : a vector containing all of the decision variables [(nX+nU+nS)*N]
                g    : a vector containing all of the constraints [(nX*2+4)*N]
                R    : a vector containing all of the reference variables [N]
                P    : a vector containing all of the parameter variables [N]
                w0   : a vector containing initial guess for the decision variables [(nX+nU+nS)*N]
                lbg  : a vector containing lower bounds on all of the constraints [(nX*2+4)*N]
                ubg  : a vector containing upper bounds on all of the constraints [(nX*2+4)*N]
                lbw  : a vector containing lower bounds on all of the decision variables [(nX+nU+nS)*N]
                ubw  : a vector containing upper bounds on all of the decision variables [(nX+nU+nS)*N]
        """

        """ Initialize the program """
        # Initialize the data structures for constraints, optimization variables and cost function
        w, w0, lbw, ubw, J, g, R, P, lbg, ubg = [], [], [], [], 0, [], [], [], [], [];

        """ Unpack important constants """
        lbg_SOC_bat = self._system_info['lbg_SOC_bat_c']; ubg_SOC_bat = self._system_info['ubg_SOC_bat_c'];
        lbg_P_wtg = self._system_info['lbg_P_wtg_c']; ubg_P_wtg = self._system_info['ubg_P_wtg_c'];
        lbg_P_bat = self._system_info['lbg_P_bat_c']; ubg_P_bat = self._system_info['ubg_P_bat_c'];

        """ Generate the whole program """
        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = ca.SX.sym('U_' + str(k), self._system_info['nU_c']); w += [Uk]; w0 += [0] * self._system_info['nU_c'];
            lbw += self._system_info['umin_c']; ubw += self._system_info['umax_c'];

            # Integrate forward in time and add the next cost objective term
            Xk_end, Qk = self.c_integrator(Xk, Uk, Sk);

            # Assuming that the next measurement is perfect and have to be prioritized by penalizing power flow slack
            if (k < 2):
                J = J + Qk + Sk[3] * self._system_info['J_cc_c_scale'];
            else:
                J = J + Qk;

            # New NLP variable for state at end of interval
            Xk = ca.SX.sym('X_' + str(k + 1), self._system_info['nX_c']);
            Sk = ca.SX.sym('S_' + str(k + 1), self._system_info['nS_c']);
            Rk = ca.SX.sym('R_' + str(k + 1), self._system_info['nR_c']);
            Pk = ca.SX.sym('P_' + str(k + 1), self._system_info['nP_c']);

            w, R, P = w + [Xk, Sk], R + [Rk], P + [Pk];
            w0 += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
            lbw += ([-np.inf] * self._system_info['nX_c'] + [0] * self._system_info['nS_c']);
            ubw += ([np.inf]  * self._system_info['nX_c'] + [np.inf] * self._system_info['nS_c']);

            # Get the power outputs of each respective power systems
            P_gtg = system_gtg(Xk[1] * self._system_info['x_c_scale'][1]);
            P_bat, SOC_bat, U_bat = system_bat(Xk[2] * self._system_info['x_c_scale'][2],
                                               Uk[1] * self._system_info['u_c_scale'][1]);
            P_wtg = system_wtg(Pk, Uk[2] * self._system_info['u_c_scale'][2])

            # Important inequality constraints
            g_P_flow = P_gtg + P_wtg + P_bat - Rk + Sk[self._system_info['nX_c']];

            # Add inequality constraint
            g   += [Xk_end - Xk, g_P_flow, SOC_bat, P_wtg, P_bat,
                    Xk + Sk[0:self._system_info['nX_c']], Xk - Sk[0:self._system_info['nX_c']]];
            lbg += ([0] * self._system_info['nX_c'] + [0] + [lbg_SOC_bat] + [lbg_P_wtg] + [lbg_P_bat] +
                    self._system_info['xmin_c'] + [-np.inf] * self._system_info['nX_c']);
            ubg += ([0] * self._system_info['nX_c'] + [0] + [ubg_SOC_bat] + [ubg_P_wtg] + [ubg_P_bat] +
                    [np.inf] * self._system_info['nX_c'] + self._system_info['xmax_c']);

        return J, w, g, w0, lbg, ubg, lbw, ubw, R, P

    def get_c_empc_optimal_references(self, x0, p, r):
        """ Compute the optimal inputs from an economic model predictive controller
            # Arguments:
                x0         : a vector containing the current initial value for controller system state [nX]
                p          : a vector containing the system parameters [nX*horizon_c]
                r          : a vector containing the references [nX*horizon_c]
            # Outputs:
                u            : a vector containing the optimal inputs [nU]
                x            : a vector containing the optimal state prediction [nX]
                iter_count   : an integer containing the amount of IPOPT iterations
                solve_status : a string containing solver status information
                w_opt        : a vector containing all of the optimal decision variables [(nX+nU+nS)*horizon_c+nX+nS]
        """

        # Iteration variable for easier access to next time step states
        n_period = self._system_info['nX_c'] + self._system_info['nS_c'] + self._system_info['nU_c'];

        # Update initial node
        w0  = ca.vertcat(x0, self._c_empc_info['w0'][self._system_info['nX_c']:]);
        lbw = ca.vertcat(x0, self._c_empc_info['lbw'][self._system_info['nX_c']:]);
        ubw = ca.vertcat(x0, self._c_empc_info['ubw'][self._system_info['nX_c']:]);
        lbg = self._c_empc_info['lbg']; ubg = self._c_empc_info['ubg'];

        # Compute optimal trajectory from nominal economic model predictive controller
        sol_c = self.c_empc(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=ca.vertcat(p,r));

        # Obtain flattened vector containing optimal state, slack and input trajectories
        w_opt = sol_c['x'].full().flatten();

        # Obtain optimal input
        u = ca.vertcat(w_opt[n_period - self._system_info['nU_c']:n_period]);

        # Obtain optimal state prediction
        x_ref = ca.vertcat(w_opt[n_period:n_period + self._system_info['nX_c']]);

        # Obtain ipopt solver information
        iter_count = self.c_empc.stats()['iter_count'];
        solve_status = self.c_empc.stats()['return_status'];

        return u, x_ref, iter_count, solve_status, w_opt

    def get_c_sempc_controller(self):
        """ Create discrete stochastic economical model predictive controller based on multiple shooting
            # Outputs:
                c_sempc      : a program instance, computing optimal input for the stochastic economic
                               model predictive controller
                c_sempc_info : a dictionary containing the stochastic economic model predictive control
                               constraints information and initial values
        """

        """ Initialize the program """
        # Initialize the data structures for constraints, optimization variables and cost function
        # w = decision variables, w0 = initial values for w, lbw = lower bounds for w, ubw = upper bounds for w
        # J = cumulative cost, g = inequality constraints, lbg = lower bounds for g, ubg = upper bounds for g
        w, w0, lbw, ubw, J, g, lbg, ubg = [], [], [], [], 0, [], [], [];

        # "Lift" initial conditions
        Xk = ca.SX.sym('X0_c', self._system_info['nX_c']);
        Sk = ca.SX.sym('S0_c', self._system_info['nS_c']);

        # Initial constraints
        w += [Xk, Sk]; w0 += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
        lbw += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
        ubw += (self._system_info['x0_c'] + [np.inf] * self._system_info['nS_c']);

        """ Generate the whole program """
        print('S-EMPC is being initialized, using multiple shooting scheme for integrating the dynamics')
        # Compute the multiple shooting branch
        J1, w1, g1, w01, lbg1, ubg1, lbw1, ubw1, R1, P1, P_cc1 = \
            self.get_c_sempc_multiple_shooting_branch(Xk, Sk, self._system_info['horizon_c']);

        # Update current program
        J, w, g, w0 = J + J1, w + w1, g + g1, w0 + w01;
        lbw, ubw, lbg, ubg = lbw + lbw1, ubw + ubw1, lbg + lbg1, ubg + ubg1;

        """ Create the stochastic economic model predictive controller instance """
        # Create a program solver instance
        c_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': ca.vertcat(*P1,*P_cc1, *R1)};
        # Create the stochastic economic model predictive controller
        c_sempc = ca.nlpsol('c_solver_sempc', 'ipopt', c_prob, self._c_IPOPT_options);
        # Save stochastic economic model predictive control constraints information and initial values
        c_sempc_info = {'w0': w0, 'lbw': lbw, 'ubw': ubw, 'lbg': lbg, 'ubg': ubg};

        return c_sempc, c_sempc_info

    def get_c_sempc_multiple_shooting_branch(self, Xk, Sk, N):
        """ Compute a whole multiple shooting branch
            # Arguments:
                Xk    : current state node [nX]
                Sk    : current slack state node [nS]
                N     : # prediction horizon for this shooting branch [1]
            # Outputs:
                J     : cumulative cost associated to this branch [1]
                w     : a vector containing all of the decision variables [(nX+nU+nS)*N]
                g     : a vector containing all of the constraints [(nX*2+4)*(c_horizon_c)+(nX*2+5)*(N-c_horizon_c)]
                R     : a vector containing all of the reference variables [N]
                P     : a vector containing all of the parameter variables [N]
                w0    : a vector containing initial guess for the decision variables [(nX+nU+nS)*N]
                lbg   : a vector containing lower bounds on all of g [(nX*2+4)*(c_horizon_c)+(nX*2+5)*(N-c_horizon_c)]
                ubg   : a vector containing upper bounds on all of g [(nX*2+4)*(c_horizon_c)+(nX*2+5)*(N-c_horizon_c)]
                lbw   : a vector containing lower bounds on all of w [(nX+nU+nS)*N]
                ubw   : a vector containing upper bounds on all of w [(nX+nU+nS)*N]
                P_cc  : a vector containing the explicit back-off for chance constraints [N]
        """

        """ Initialize the program """
        # Initialize the data structures for constraints, optimization variables and cost function
        w, w0, lbw, ubw, J, g, R, P, lbg, ubg, P_cc = [], [], [], [], 0, [], [], [], [], [], [];

        """ Unpack important constants """
        lbg_SOC_bat = self._system_info['lbg_SOC_bat_c']; ubg_SOC_bat = self._system_info['ubg_SOC_bat_c'];
        lbg_P_wtg = self._system_info['lbg_P_wtg_c']; ubg_P_wtg = self._system_info['ubg_P_wtg_c'];
        lbg_P_bat = self._system_info['lbg_P_bat_c']; ubg_P_bat = self._system_info['ubg_P_bat_c'];
        lbg_P_error_gtg_bat = self._system_info['lbg_P_error_gtg_bat_c'];
        ubg_P_error_gtg_bat = self._system_info['ubg_P_error_gtg_bat_c'];

        """ Generate the whole program """
        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = ca.SX.sym('U_' + str(k), self._system_info['nU_c']); w += [Uk]; w0 += [0] * self._system_info['nU_c'];
            lbw += self._system_info['umin_c']; ubw += self._system_info['umax_c'];

            # Integrate till the end of the interval
            Xk_end, Qk = self.c_integrator(Xk, Uk, Sk);

            # Assuming that the next measurement is perfect and have to be prioritized by penalizing power flow slack
            if(k<2):
                J = J + Qk + Sk[3] * self._system_info['J_cc_c_scale'];
            else:
                J = J + Qk;

            # New NLP variable for state at end of interval
            Xk = ca.SX.sym('X_' + str(k + 1), self._system_info['nX_c']);
            Sk = ca.SX.sym('S_' + str(k + 1), self._system_info['nS_c']);
            Rk = ca.SX.sym('R_' + str(k + 1), self._system_info['nR_c']);
            Pk = ca.SX.sym('P_' + str(k + 1), self._system_info['nP_cc_c']);

            w, R, P, P_cc = w + [Xk, Sk], R + [Rk], P + [Pk[0]],P_cc + [Pk[1]];
            w0 += (self._system_info['x0_c'] + [0] * self._system_info['nS_c']);
            lbw += ([-np.inf] * self._system_info['nX_c'] + [0] * self._system_info['nS_c']);
            ubw += ([np.inf]  * self._system_info['nX_c'] + [np.inf] * self._system_info['nS_c']);

            # Get the power outputs of each respective power systems
            P_gtg = system_gtg(Xk[1]*self._system_info['x_c_scale'][1]);
            P_bat, SOC_bat, U_bat = system_bat(Xk[2] * self._system_info['x_c_scale'][2],
                                                   Uk[1] * self._system_info['u_c_scale'][1]);
            P_wtg = system_wtg(Pk[0],Uk[2]*self._system_info['u_c_scale'][2]);

            # Add inequality constraint
            if (k < self._system_info['c_horizon_c']): # Assuming that the next predictions are perfect

                # Important inequality constraints
                g_P_flow = P_gtg + P_wtg + P_bat - Rk + Sk[self._system_info['nX_c']];

                # Add constraints in the nominal domain
                g += [Xk_end - Xk, g_P_flow, SOC_bat, P_wtg, P_bat,
                      Xk + Sk[0:self._system_info['nX_c']], Xk - Sk[0:self._system_info['nX_c']]];
                lbg += ([0] * self._system_info['nX_c'] + [0] + [lbg_SOC_bat] + [lbg_P_wtg] + [lbg_P_bat] +
                        self._system_info['xmin_c'] + [-np.inf] * self._system_info['nX_c']);
                ubg += ([0] * self._system_info['nX_c'] + [0] + [ubg_SOC_bat] + [ubg_P_wtg] + [ubg_P_bat] +
                        [np.inf] * self._system_info['nX_c'] + self._system_info['xmax_c']);

            else: # Assuming that the next predictions are uncertain and have to be handled in the Lamperti domain

                # Get the explicit backoff for chance constraints in the Lamperti domain
                P_wtg_backoff_L = Pk[1];

                # Transform the power demand, gas power, and battery power to the Lamperti domain
                P_error_gtg_bat = (Rk - P_bat - P_gtg)/self._system_info['P_error_gtg_bat_c_scale']; # Scale the power error without wind
                P_error_gtg_bat_L =  self.get_c_P_L(P_error_gtg_bat); # Lamperti domain transform the error without wind

                # Important inequality constraints in the Lamperti domain
                g_P_flow_L = P_error_gtg_bat_L - Sk[3] - P_wtg_backoff_L;

                # Add constraints in the Lamperti domain
                g += [Xk_end-Xk, g_P_flow_L, SOC_bat, P_wtg, P_bat, P_error_gtg_bat,
                      Xk+Sk[0:self._system_info['nX_c']], Xk-Sk[0:self._system_info['nX_c']]];
                lbg += ([0] * self._system_info['nX_c'] + [0] + [lbg_SOC_bat] + [lbg_P_wtg] + [lbg_P_bat] +
                        [lbg_P_error_gtg_bat] + self._system_info['xmin_c'] + [-np.inf] * self._system_info['nX_c']);
                ubg += ([0] * self._system_info['nX_c'] + [0] + [ubg_SOC_bat] + [ubg_P_wtg] + [ubg_P_bat] +
                        [ubg_P_error_gtg_bat] + [np.inf] * self._system_info['nX_c'] + self._system_info['xmax_c']);

        return J, w, g, w0, lbg, ubg, lbw, ubw, R, P, P_cc

    def get_c_sempc_optimal_references(self, x0, p, p_cc, r):
        """ Compute the optimal inputs from a stochastic economic model predictive controller
            # Arguments:
                x0           : a vector containing the current initial value for system state [nX]
                p            : a vector containing the system parameters [nX*horizon_c]
                p_cc         : a vector containing the explicit back-off for chance constraints [nX*horizon_c]
                r            : a vector containing the references [nX*horizon_c]
            # Outputs:
                u            : a vector containing the optimal inputs [nU]
                x            : a vector containing the optimal state prediction [nX]
                iter_count   : an integer containing the amount of IPOPT iterations
                solve_status : a string containing solver status information
                w_opt        : a vector containing all of the optimal decision variables [(nX+nU+nS)*horizon_c+nX+nS]
        """

        # Iteration variable for easier access to next time step states
        n_period = self._system_info['nX_c'] + self._system_info['nS_c'] + self._system_info['nU_c'];

        # Update initial node
        w0  = ca.vertcat(x0, self._c_sempc_info['w0'][self._system_info['nX_c']:]);
        lbw = ca.vertcat(x0, self._c_sempc_info['lbw'][self._system_info['nX_c']:]);
        ubw = ca.vertcat(x0, self._c_sempc_info['ubw'][self._system_info['nX_c']:]);
        lbg = self._c_sempc_info['lbg']; ubg = self._c_sempc_info['ubg'];

        # Compute optimal trajectory from stochastic-constrained model predictive controller
        sol_c = self.c_sempc(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=ca.vertcat(p,p_cc, r));

        # Obtain flattened vector containing optimal state, slack and input trajectories
        w_opt = sol_c['x'].full().flatten();

        # Obtain optimal input
        u = ca.vertcat(w_opt[n_period - self._system_info['nU_c']:n_period]);

        # Obtain optimal state prediction
        x_ref = ca.vertcat(w_opt[n_period:n_period + self._system_info['nX_c']]);

        # Obtain ipopt solver information
        iter_count = self.c_sempc.stats()['iter_count'];
        solve_status = self.c_sempc.stats()['return_status'];

        return u, x_ref, iter_count, solve_status, w_opt

    def get_c_P_L(self,P):
        """ Get the power in the Lamperti domain using linear approximation of the actual trasnformation - arcsin(x/2 -1)
            As an approximation we linearise the transformation.
            # Arguments:
                int/sym P   : Power P in the nominal domain (-)
            # Outputs:
                int/sym P_L : Power P in the Lamperti domain [-]
        """
        return (0.75*P - 1.5)

    def get_c_P(self,v_L,P_L):
        """ Get the power in the nominal domain by forcing the power onto a sigmoid
            # Arguments:
                int/sym v_L  : average wind speed v in the Lamperti domain (-)
                int/sym P_L  : Power P in the Lamperti domain (-)
            # Outputs:
                int/sym P    : Power P in the nominal domain [-]
        """
        P = (0.5+0.5*np.tanh(self._system_info['gamma_d'][0]*(v_L**2/4-self._system_info['gamma_d'][1])))\
            *0.5*(1+np.sin(P_L))*self._system_info['u_c_scale'][2]
        return P

    def get_c_P_error(self,P1,P2):
        """ Get the power difference error in the nominal domain
            # Arguments:
                int/sym P1      : Power P in the Lamperti domain [-]
                int/sym P2      : Power P in the Lamperti domain [-]
            # Outputs:
                int/sym P_error : Power difference P_error in the nominal domain [-]
        """
        P_error = np.abs((0.5*(1+np.sin(P1))-0.5*(1+np.sin(P2)))*self._system_info['u_c_scale'][2])
        return P_error

    def get_c_P_simple(self,P_L):
        """ Get the power in the nominal domain
            # Arguments:
                int/sym P_L  : Power P in the Lamperti domain [-]
            # Outputs:
                int/sym P    : Power P in the nominal domain [MW]
        """
        P = (0.5*(1+np.sin(P_L))*self._system_info['u_c_scale'][2])
        return P

    def get_c_v(self,v_L):
        """ Get the average wind speed in the nominal domain
            # Arguments:
                int/sym v_L  : average wind speed v in the Lamperti domain (-)
            # Outputs:
                int/sym v    : average wind speed v in the nominal domain [m/s^2]
        """
        v = v_L**2/4
        return v

    def get_d_system_estimate_kf(self, y, t_grid, x0_kf_L, P0_kf_L, _args):
        """ Compute system estimates/covariance for the forecast method with a continous-discrete extended Kalman filter
            # Arguments:
                y:            : The new obsveration at time [nY]
                t_grid:       : An array, [tk-1, tk], which is the time step from the previous time step
                x0_kf_L       : The filtered system state at time tk-1 [nX,1]
                P0_kf_L       : The filtered system covariance at time tk-1 [nX,nX]
                _args         : A touple with additional arguments to the integration function, (NWP_k, diff(NPW)_k, pars)

            # Outputs:
                x1_kf_L       : Filtered next step state estimate in the Lamperti domain [nX,1]
                P1_kf_L       : Filtered next step state covariance estimates in the Lamperti domain [nX,nX]
        """

        """ Init initialization """
        # observation covariance matrix
        R = np.diag((self._system_info['sigma_y_kf_d'][0], self._system_info['sigma_y_kf_d'][1]))

        """ Prediction step """
        # solve ODEs to obtain xpred, Ppred
        w0_L = np.concatenate((x0_kf_L, P0_kf_L.reshape(16, )))
        res = odeint(d_f_L, w0_L, t_grid, args=_args)  # args=(p[k], dp[k], pars)
        xpred = res[1, :4].copy().reshape((4, 1))
        Ppred = res[1, 4:].reshape(4, 4).copy()

        """ Update step """
        z = [xpred[0, 0], xpred[2, 0]]
        # one-step prediction of the measurement
        Ck = self.d_dhdx(z).full()  # linearised observation equations
        ypred = self.d_h(z).full()

        """ Correction step """
        # estimated process noice and Kalman gain
        Rk = Ck @ Ppred @ Ck.T + R
        K = Ppred @ Ck.T @ np.linalg.inv(Rk)

        # filtered estimates
        I = np.eye(Ppred.shape[0]);
        x1_kf_L = xpred + K @ (y - ypred);
        P1_kf_L = (I - K @ Ck) @ Ppred @ (I - K @ Ck).T + K @ Rk @ K.T;

        return x1_kf_L, P1_kf_L

    def get_d_forecasts_sode(self,x0, P0, _pt, _NWP):
        """ This function forecasts the wind power production together with its density (given by a mean and
            covariance). The computations are performed in the Lamperti domain and the densities are
            returned in the Lamperti domain as well with their uncertainty.
            # Arguments:
                x0:           The initial conditions in the Lamperti domain, numpy array with shape (4,1)
                P0:           The initial covariance of the system at time t0, numpy array of size (4,4)
                pt:           A touple with the desired density percentiles, (pt1, pt2, ..., ptn)
                _NWP:         The numerical weather predictions, numpy array with shape (N,5) where columns are (time, wind, NWP, diff(NWP), power)
                pars:         Model parameters

            # Outputs:
                qt:           The quantile values of the forecasted normal density of the variables for each time in array t, array of size (m,n)
                mean:         The mean at each point in time, numpy array with size (m,4)
                var:          The covariance matrix at each point in time, numpy arry with size (m,4,4)
        """


        qt = ss.norm.ppf(_pt, loc=0, scale=1);
        N = _NWP.shape[0];
        m = len(_pt);

        # find indicies for wind speed forecasts
        # tt = _NWP[:,0]
        dt = _NWP[:2, 0];
        p = _NWP[:, 2];
        dp = _NWP[:, 3];

        # initialise arrays
        w0 = np.concatenate((x0.reshape(4, ),
                             P0.reshape(4 * 4, )))
        mean = np.zeros((N, 4))
        var = np.zeros((N, 4, 4))

        # compute densities for each point in time
        for k in range(N):
            # tk = [tt[k], tt[k+1]]
            res = odeint(d_f_L, w0, dt, args=(p[k], dp[k],  self._system_info['parameters_d']))
            w0 = res[1, :].copy()
            mean[k, :] = res[1, :4].copy()
            var[k, :, :] = res[1, 4:].reshape(4, 4).copy()

        # compute quantiles
        powerquantiles = np.zeros((N, m))
        for i in range(m):
            powerquantiles[:, i] = mean[:, 2] + qt[i] * np.sqrt(var[:, 2, 2])  # Lamperti domain

        return powerquantiles, mean, var

    def get_d_forecast_predictions(self,M, N,P0_kf_L,wind_data):
        """ Compute advanced forecasts of average wind and wind power output with Kalman filter and Nonlinear
            Stochastic Differential Equations with their uncertainty.
            # Arguments:
                M                   : an integer indicating the current time slice in terms of the wind_data [1]
                N                   : an integer indicating the horizon for which the forecast methods predicts for [1]
                P0_kf_L             : a matrix containing the previous time step system covariances [nX,nX]
                wind_data           : a dictionary containing numerical wind data (meteorological, observed ..)
            # Outputs:
                x1_kf_L             : a vector containing the next time step system estimates [nX]
                P1_kf_L             : a matrix containing the next time step system covariances [nX,nX]
                x_mean_forecast_L   : a matrix containing the mean of the system state forecasts [N,nX]
                x_var_forecast_L    : a matrix containing the variance of the system state forecasts [N,nX, nX]
                quantile_forecast_L : a vector containing all of the optimal decision variables [N,2]
        """

        ''' Compute next system state and covariance using a continous-discrete Kalman filter '''
        # Compute the outputs in the Lamperti domain given current measurements M
        y1_L = [np.log(2 * np.sqrt(wind_data['wind_perfect'][M]))]; # Compute the first output from current observation
        y2_L = [logit(wind_data['P_wind_perfect'][M])]; # Compute the second output from current observation
        y_kf_L = np.array([y1_L, y2_L]); # Collect the measurements

        t_grid_kf = [0, 1/12]  # Time grid for the Kalman filter

        # Compute the previous estimates based on the previous measurements M-1
        x0_1_L = 2 * np.sqrt(wind_data['wind_perfect'][M - 1]); # Compute the first state from previous observation
        x0_2_L = 0; # Compute the second state from previous observation
        x0_3_L = wind_data['P_wind_perfect'][M - 1]; # Compute the third state from previous observation
        x0_4_L = 0; # Compute the fourth state from previous observation
        x0_kf_L = np.array([x0_1_L, x0_2_L, x0_3_L,x0_4_L]) # Collect the system states

        # Arguments to the integration function
        args = (wind_data['wind_imperfect'][M - 1], wind_data['wind_pred'][M - 1], self._system_info['parameters_d']);

        # Compute the next time step system state and covariance
        x1_kf_L, P1_kf_L = self.get_d_system_estimate_kf(y_kf_L, t_grid_kf, x0_kf_L, P0_kf_L, args);

        ''' Compute forecasts of wind using stochastic differential equations '''
        # Specify the quantiles for which we want to forecast for
        quantiles = (0.05, 0.95);

        # Arguments to the sode function
        args = wind_data['wind_all'][M:M + N, :];

        # Compute the forecasts using stochastic differential equations
        qt_forecast_L, x_mean_forecast_L, x_var_forecast_L = self.get_d_forecasts_sode(x1_kf_L, P1_kf_L, quantiles, args)


        return x1_kf_L,P1_kf_L,x_mean_forecast_L,x_var_forecast_L, qt_forecast_L

    def get_d_forecasts_sode_simple(self,x0, P0, _pt, _NWP):
        """ This function forecasts the wind power production together with its density (given by a mean and
            covariance). The computations are performed in the Lamperti domain and the densities are
            returned in the Lamperti domain as well with no uncertainty.
            # Arguments:
                x0:           The initial conditions in the Lamperti domain, numpy array with shape (4,1)
                P0:           The initial covariance of the system at time t0, numpy array of size (4,4)
                pt:           A touple with the desired density percentiles, (pt1, pt2, ..., ptn)
                _NWP:         The numerical weather predictions, numpy array with shape (N,5) where columns are (time, wind, NWP, diff(NWP), power)
                pars:         Model parameters

            # Outputs:
                qt:           The quantile values of the forecasted normal density of the variables for each time in array t, array of size (m,n)
                mean:         The mean at each point in time, numpy array with size (m,4)
                var:          The covariance matrix at each point in time, numpy arry with size (m,4,4)
        """


        qt = ss.norm.ppf(_pt, loc=0, scale=1)
        N = _NWP.shape[0]
        m = len(_pt)

        # find indicies for wind speed forecasts
        # tt = _NWP[:,0]
        dt = _NWP[:2, 0]
        p = _NWP[:, 2]
        dp = _NWP[:, 3]

        # initialise arrays
        w0 = np.concatenate((x0.reshape(4, ),
                             P0.reshape(4 * 4, )))
        mean = np.zeros((N, 4))
        with alive_bar(N) as bar:  # declare your expected total
            # compute densities for each point in time
            for k in range(N):
                # tk = [tt[k], tt[k+1]]
                res = odeint(d_f_L, w0, dt, args=(p[k], dp[k],  self._system_info['parameters_d']))
                w0 = res[1, :].copy()
                mean[k, :] = res[1, :4].copy()
                bar()  # call after consuming one item for visual bar

        return mean

    def get_d_forecast_predictions_simple(self,M, N,P0_kf_L,wind_data):
        """ Compute advanced forecasts of average wind and wind power output with Kalman filter and Nonlinear
            Stochastic Differential Equations with no uncertainty.
            # Arguments:
                M                   : an integer indicating the current time slice in terms of the wind_data [1]
                N                   : an integer indicating the horizon for which the forecast methods predicts for [1]
                P0_kf_L             : a matrix containing the previous time step system covariances [nX,nX]
                wind_data           : a dictionary containing numerical wind data (meteorological, observed ..)
            # Outputs:
                x1_kf_L             : a vector containing the next time step system estimates [nX]
                P1_kf_L             : a matrix containing the next time step system covariances [nX,nX]
                x_mean_forecast_L   : a matrix containing the mean of the system state forecasts [N,nX]
                x_var_forecast_L    : a matrix containing the variance of the system state forecasts [N,nX, nX]
                quantile_forecast_L : a vector containing all of the optimal decision variables [N,2]
        """

        ''' Compute next system state and covariance using a continous-discrete Kalman filter '''
        # Compute the outputs in the Lamperti domain given current measurements M
        y1_L = [np.log(2 * np.sqrt(wind_data['wind_perfect'][M]))]; # Compute the first output from current observation
        y2_L = [logit(wind_data['P_wind_perfect'][M])]; # Compute the second output from current observation
        y_kf_L = np.array([y1_L, y2_L]); # Collect the measurements

        t_grid_kf = [0, 1/12]  # Time grid for the Kalman filter

        # Compute the previous estimates based on the previous measurements M-1
        x0_1_L = 2 * np.sqrt(wind_data['wind_perfect'][M - 1]); # Compute the first state from previous observation
        x0_2_L = 0; # Compute the second state from previous observation
        x0_3_L = wind_data['P_wind_perfect'][M - 1]; # Compute the third state from previous observation
        x0_4_L = 0; # Compute the fourth state from previous observation
        x0_kf_L = np.array([x0_1_L, x0_2_L, x0_3_L,x0_4_L]) # Collect the system states

        # Arguments to the integration function
        args = (wind_data['wind_imperfect'][M - 1], wind_data['wind_pred'][M - 1], self._system_info['parameters_d']);

        # Compute the next time step system state and covariance
        x1_kf_L, P1_kf_L = self.get_d_system_estimate_kf(y_kf_L, t_grid_kf, x0_kf_L, P0_kf_L, args);

        ''' Compute forecasts of wind using stochastic differential equations '''
        # Specify the quantiles for which we want to forecast for
        quantiles = (0.05, 0.95);

        # Arguments to the sode function
        args = wind_data['wind_all'][M:M + N, :];

        # Compute the forecasts using stochastic differential equations
        x_mean_forecast_L = self.get_d_forecasts_sode_simple(x1_kf_L, P1_kf_L, quantiles, args)


        return x1_kf_L,P1_kf_L,x_mean_forecast_L