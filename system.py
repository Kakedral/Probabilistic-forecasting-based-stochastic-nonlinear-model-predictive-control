""" Overview """
# Script for system equation and cost function for the different power systems in the offshore
# hybrid power system. Also incudes simulation of plant with controller.
# Copyright (c) 2021, Kiet Tuan Hoang/ Christian Ankerstjerne Thilker
# Last edited: 27.07.2022

""" Reference """
# Gas turbine model - M.Nagpal, A.Moshref, G.K.Morison, and P.Kundur. Experience with testing and modeling of gasturbines
#                     In 2001 IEEE Power Engineering Society Winter Meeting. Conference Proceedings (Cat.No.01CH37194),
#                     volume  2,  pages  652–656  vol.2,  2001.doi: 10.1109/PESW.2001.916931
# Battery model - C.M. Shepherd.  Design of primary and secondary cells ii.an equation describing battery discharge.
#                 Journal of the Electrochemical Society 112.7, pages 657 – 667, 1965

''' Abbreviations '''
# General model predictive control functions are abbreviated with _mpc
# Economic model predictive control functions are abbreviated with _empc
# Stochastic model predictive control functions are abbreviated with _sempc
# Kalman filter equations and functions are abbreviated with _kf
# Stochastic differential equations are abbreviated with _sode
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
Dependencies information (modules)
config  : a library for utility functions and configurations
method  : a library for controller and forecast functions

* config_config_color_map          : a dictionary containing various colormaps from NTNU and RWTH Aachen
* config_config_sublist_generator  : a function to partition a list into multiple sublist for receeding horizon control
* get_c_P_L                        : a function to compute the Lamperti transformation of power
"""

''' Module imports '''
from config import config_color_map
from config import config_sublist_generator

"""
Dependencies information (libraries)
* os             : a general library for pathing w.r.t folders
* numpy          : a general library for numerical operations
* scipy          : a general library for numerical integration and formulas
* pandas         : a library for retrieving numerical data from .xlsx and .csv files
* casadi         : a general library for symbolic operations
* datetime       : a library for time and space for documentation
* matplotlib     : a general library for plotting results 
* alive_progress : a general library for creating a progress bar
"""

''' Imports '''
import os

import numpy as np
import pandas as pd
import casadi as ca
import pickle as pc

import scipy.stats as ss

import matplotlib.pyplot as plt
from alive_progress import alive_bar

""" System dynamics and auxiliary subsystem functions for the simulator, the controllers, and the forecast method """
def system_OHPS():
    """ Initialize Offshore Hybrid Power System constants, functions, controllers, and forecast method
        # Arguments:
            -
        # Outputs:
            system_info : a dictionary with all of the relevant system information
                          such as system dimensions, sampling time, process and measurment
                          disturbance standard deviation, initial values ...
            system_var  : a dictionary with all of the symbolic variables
            c_f         : state transition function - dxdt = f(x, u, p) for the controller
            c_h         : symbolic measurement function - y = h(x, u, p) for the controller
            c_L         : combined symbolic cost function and state transition function for the controller
            d_f         : state transition function - dxdt = f(x, u, p) for the disturbance forecast method
            d_h         : symbolic measurement function - y = h(x, u, p) for the disturbance forecast method
   """

    """ Initial system definitions (The control model = (system model - wind dynamics) as wind is in forecast method) """
    dt = 150;  # System time constant [1]

    nX = 3;  # System state vector length [1]
    nU = 3;  # System input vector length [1]
    nP = 1;  # System parameter vector length [1]
    nY = 2;  # System output vector length [1]

    x0 = [0.001, 0.001, 0.001];  # System initial state vector [nX]
    u0 = [0.001, 0.001]; # System initial input vector [nU]
    xmin_system = [0, 0  , -1.1];  # Real life system state minimum values [nX]
    xmax_system = [1, 1.2,  1.2];  # Real life system state maximum values [nX]

    x_scale = [1, 100, 9360];  # System state scaling vector [nX]
    y_scale = [100, 1];  # System output scaling vector [nY]
    u_scale = [1, 200, 88];  # System input scaling vector [nU]

    """ Initial controller definitions """
    horizon_c = 120;  # Prediction horizon of the controller [1]
    discretization_c = 120;  # Control discretization of the high level controller [1]
    c_horizon_c = 4; # A horizon for which the controller can assume perfect forecast [1]
    IPOPT_solver = 'mumps'; # The linear solver for which steps are computed [5]

    nX_c = 3;  # Controller system state vector length [1]
    nU_c = 3;  # Controller system input vector length [1]
    nP_c = 1;  # Controller system parameter vector length [1]
    nP_cc_c = 2;  # Controller system parameter vector length for chance constrained problems [1]
    nY_c = 2;  # Controller system output vector length [1]
    nS_c = 4;  # Controller system slack vector length [1]
    nR_c = 1;  # Controller system reference vector length [1]

    xmin_c = [0, 0, -1];  # Controller system state minimum values [nX_c]
    xmax_c = [1, 1,  1];  # Controller system state maximum values [nX_c]
    umin_c = [0, -1,  0];  # Controller system input mimimum values [nU_c]
    umax_c = [1,  1,  1];  # Controller system input maximum values [nU_c]

    x0_c = [0.001, 0.001, 0.001];  # Controller system initial state vector [nX_c]
    u0_c = [0.001, 0.001];  # Controller system initial input vector [nU_c]

    x_c_scale = [1, 100, 9360];  # Controller system state scaling vector [nX_c]
    y_c_scale = [100, 1];  # Controller system output scaling vector [nY_c]
    u_c_scale = [1, 200, 88];  # Controller system input scaling vector [nU_c]

    P_error_gtg_bat_c_scale = 25; # Controller system power demand scaling constant given no wind power [1]

    lbg_P_error_gtg_bat_c = 0; # Lower bound on normalized power demand error from using gas turbines and batteries [1]
    ubg_P_error_gtg_bat_c = 4; # Upper bound on normalized power demand error from using gas turbines and batteries [1]
    lbg_SOC_bat_c = 0.1; # Lower bound on the battery state of charge [1]
    ubg_SOC_bat_c = 1; # Upper bound on the battery state of charge [1]
    lbg_P_wtg_c = 0; # Lower bound on the wind turbine generator power output [1]
    ubg_P_wtg_c = u_scale[2]; # Upper bound on the wind turbine generator power output (motivated by Hywind Tampen) [1]
    lbg_P_bat_c = -80; # Lower bound on the battery power output [1]
    ubg_P_bat_c = 80; # Upper bound on the battery power output [1]

    ''' Stochastic constraint constants '''
    J_cc_c_scale = 80000; # Constant to penalize and prioritse the first time slice [1]
    epsilon_cc_c = 0.075; # Tuning constants for quantile function based chance-constraints [1]
    epsilon_cc2_c = 0.2898; # Tuning constants for quantile function based chance-constraints [1]
    epsilon_cc3_c = 0.266;#0.76524; # Tuning constants for quantile function based chance-constraints, 0.775 -> non-scaled [1]

    """ Initial forecasting definitions """
    dt_d = 2; # Sampling time of the disturbance forecast method [1]
    P0_kf_d = np.diag([0.4, 4, 0.2, 1.7]); # Initial covariance for the disturbance estimation in the Kalman filter

    nX_d = 4;  # Disturbance forecast method system state vector length [1]
    nY_d = 2;  # Disturbance forecast method system output vector length [1]

    """ These are all derived through maximum likelehood """
    theta_v_d = 0.21;  # 0.17 [1]
    theta_R_d = 6.16; # [1]
    theta_P_d = 2.88; # [1]
    theta_Q_d = 0.29; # [1]

    rho_d = 0.15; # [1]
    mu_d = 1.19; # [1]

    gamma1_d = 0.900; # [1]
    gamma2_d = 4.69; # [1]

    xi1_d = 0.46; # [1]
    xi2_d = 9.47; # [1]
    xi3_d = 0.99; # [1]

    sigma_v_d = np.exp(-4.11); # -0.716
    sigma_R_d = np.exp(1.88);  # np.exp(1.116)
    sigma_P_d = np.exp(-2.77); # [1]
    sigma_Q_d = np.exp(0.14); # [1]

    sigma_y1_d = np.exp(-7.000); # [1]
    sigma_y2_d = np.exp(-4.640); # [1]

    # Collect some of the forecast parameters into one list
    theta_d = [theta_v_d, theta_R_d, theta_P_d, theta_Q_d];
    sigma_x_kf_d = [sigma_v_d, sigma_R_d, sigma_P_d, sigma_Q_d];
    sigma_y_kf_d = [sigma_y1_d, sigma_y2_d*2.5];
    xi_d    = [xi1_d, xi2_d, xi3_d];
    gamma_d = [gamma1_d,gamma2_d];

    # Collect the Kalman filter parameters into one list
    parameters_d = [theta_v_d, theta_R_d, theta_P_d, theta_Q_d, rho_d, mu_d,
                    gamma1_d, gamma2_d, xi1_d, xi2_d, xi3_d, sigma_v_d, sigma_R_d, sigma_P_d, sigma_Q_d]

    """ Simulation parameters (general) """
    rand_seed = 1;  # Seed for reproducibility [1]
    dt_v = 12; # Sampling time of wind power data, which here is every 5 minute (5/60) [1]
    variable_power = True; # A boolean, used to indicate whether power demand is constant or follows meteorological forecasts [1]
    variable_power_scale = 0.8; # A variable to scale the power demand given meteorological forecasts and max gas power ouput [1]

    """ Simulation parameters (1 day case study) """
    t_start = 1912; # Start time for 1 day simulations [1]

    """ System specifications """
    system_info = {
        # General system hyperparameters
        'nX': nX, 'nU': nU, 'nP' : nP, 'nY': nY,  'dt': dt,
        'x0': x0, 'u0': u0, 'xmin_system': xmin_system, 'xmax_system': xmax_system,
        'y_scale': y_scale, 'x_scale': x_scale, 'u_scale': u_scale,

        # General control hyperparameters
        'horizon_c': horizon_c, 'discretization_c': discretization_c, 'c_horizon_c': c_horizon_c,

        'nX_c': nX_c, 'nU_c': nU_c, 'nP_c': nP_c, 'nP_cc_c': nP_cc_c, 'nY_c': nY_c, 'nS_c': nS_c, 'nR_c': nR_c,
        'x0_c': x0_c, 'u0_c': u0_c, 'xmin_c': xmin_c, 'xmax_c': xmax_c, 'umin_c': umin_c, 'umax_c': umax_c,
        'y_c_scale': y_c_scale, 'x_c_scale': x_c_scale, 'u_c_scale': u_c_scale,
        'P_error_gtg_bat_c_scale': P_error_gtg_bat_c_scale, 'IPOPT_solver': IPOPT_solver,

        'lbg_P_error_gtg_bat_c': lbg_P_error_gtg_bat_c, 'ubg_P_error_gtg_bat_c': ubg_P_error_gtg_bat_c,
        'lbg_SOC_bat_c': lbg_SOC_bat_c, 'ubg_SOC_bat_c': ubg_SOC_bat_c,
        'lbg_P_wtg_c': lbg_P_wtg_c, 'ubg_P_wtg_c': ubg_P_wtg_c,
        'lbg_P_bat_c': lbg_P_bat_c, 'ubg_P_bat_c': ubg_P_bat_c,

        # Stochastic control hyperparameters
        'J_cc_c_scale': J_cc_c_scale,
        'epsilon_cc_c':epsilon_cc_c, 'epsilon_cc2_c':epsilon_cc2_c, 'epsilon_cc3_c':epsilon_cc3_c,

        # Forecasting hyperparameters
        'dt_d':dt_d, 'P0_kf_d' :P0_kf_d, 'parameters_d':parameters_d, 'nX_d': nX_d, 'nY_d':nY_d,
        'theta_d': theta_d, 'sigma_x_kf_d': sigma_x_kf_d, 'sigma_y_kf_d':sigma_y_kf_d, 'xi_d':xi_d,
        'gamma_d':gamma_d, 'rho_d':rho_d, 'mu_d':mu_d,

        # Simulation hyperparameters
        'rand_seed': rand_seed, 't_start':t_start ,
        'variable_power':variable_power, 'variable_power_scale' : variable_power_scale,'dt_v': dt_v
    }

    # System symbolic variables
    x = ca.SX.sym('x', system_info['nX'], 1); u = ca.SX.sym('u', system_info['nU'], 1);
    y = ca.SX.sym('y', system_info['nY'], 1); p = ca.SX.sym('p', system_info['nP'], 1);

    # Controller symbolic variables
    x_c = ca.SX.sym('x_c', system_info['nX_c'], 1); u_c = ca.SX.sym('u_c', system_info['nU_c'], 1);
    y_c = ca.SX.sym('y_c', system_info['nY_c'], 1); r_c = ca.SX.sym('r_c', system_info['nR_c'], 1);
    s_c = ca.SX.sym('s_c', system_info['nS_c'], 1); p_c = ca.SX.sym('p_c', system_info['nP_c'], 1);
    p_cc_c = ca.SX.sym('p_cc_c', system_info['nP_cc_c'], 1)

    # Forecast method symbolic variables
    x_d = ca.SX.sym('x_d', system_info['nX_d'], 1); y_d = ca.SX.sym('y_d', system_info['nY_d'], 1);

    # Save symbolic variables
    system_var = {'x_c': x_c, 'u_c': u_c, 'p_c': p_c, 'y_c': y_c, 's_c': s_c, 'r_c': r_c, 'p_cc_c': p_cc_c,
                  'x': x, 'u': u, 'p': p, 'y': y, 'x_d': x_d, 'y_d': y_d}

    """ Offshore Hybrid Power System dynamics for the simulator and controller """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ''' Gas turbine generator system (linear), gtg = gas turbine generator '''
    V_gtg = x_c[0] * system_info['x_c_scale'][0];  # Gas turbine state 1 - Fuel flow [pu]
    P_gtg = system_gtg(x_c[1] * system_info['x_c_scale'][1]);  # Gas turbine state 2 - Power flow [MW]
    T_gtg = u_c[0] * system_info['u_c_scale'][0];  # Gas turbine input - Gas throttle [pu]

    # Gas turbine constants
    tau_V_gtg, tau_P_gtg = 5, 0.1;  # Gas turbine constant parameters - state time constants [s]

    ''' Battery system '''
    Q_bat = x_c[2] * system_info['x_c_scale'][2];  # Battery system state - Battery charge [Q]
    I_bat = u_c[1] * system_info['u_c_scale'][1];  # Battery system input - Battery current [I]

    # Battery additional variables (Power output [MW], State of Charge [-], Voltage [V])
    P_bat, SOC_bat, U_bat = system_bat(Q_bat, I_bat);

    ''' Total system '''
    dV_gtg_dt = ((T_gtg - V_gtg) / tau_V_gtg);
    dP_gtg_dt = ((V_gtg * system_info['x_c_scale'][1] - P_gtg) / tau_P_gtg);
    dQ_bat = I_bat;

    # Symbolic controller and system ordinary differential equations
    dxdt_c = ca.vertcat(dV_gtg_dt / system_info['x_c_scale'][0], dP_gtg_dt / system_info['x_c_scale'][1],
                      dQ_bat / system_info['x_c_scale'][2]);
    c_f = ca.Function('f_c', [x_c, u_c], [dxdt_c]);

    # Symbolic controller and system output equation
    y_c = ca.vertcat(P_gtg / system_info['y_c_scale'][0], SOC_bat / system_info['y_c_scale'][1]);
    c_h = ca.Function('h_c', [x_c, u_c], [y_c]);

    """ Control objective of the Offshore Hybrid Power System """
    Q_c, R_c, Qs_c = np.diag([-0.1, 30]), np.diag([1, 10, 1]), np.diag([1, 1, 1, 50]); # Controller tuning matrices

    # Symbolic controller cost function
    cost_c = Q_c[0, 0] * SOC_bat / system_info['y_c_scale'][1] + u_c.T @ R_c @ u_c + s_c.T @ Qs_c @ s_c + \
             Q_c[1, 1] * P_gtg / system_info['y_c_scale'][0]
    c_L = ca.Function('c_L', [x_c, u_c, s_c], [dxdt_c, cost_c])

    """ Average wind speed and wind power dynamics for the disturbance forecast method """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ''' Measurement function (nonlinear), wtg = wind turbine generator'''
    v_wind = y_d[0]; # Average wind speed at the wind turbines [m/s^2]
    P_wtg  = y_d[1]; # Net power output (observed) at the wind turbine generator [MW]
    p_v_d  = gamma1_d * (v_wind ** 2 - 4 * gamma2_d) / 4;  # Intermediary variable

    # Total differentiated measurement system
    dhdP_wtg = 4 * np.cosh(p_v_d) * np.cos(P_wtg) / ((2 * np.sin(P_wtg) + 2 + np.cos(P_wtg) ** 2) * np.cosh(p_v_d) -
               2 * np.sinh(p_v_d) * np.sin(P_wtg) - 2 * np.sinh(p_v_d) + np.sinh(p_v_d) * np.cos(P_wtg) ** 2);
    dhdv     = 2 * gamma1_d * v_wind * (np.sinh(p_v_d) - 1 * np.cosh(p_v_d)) / (
               (np.sin(P_wtg) - 3) * np.cosh(p_v_d) + (1 + np.sin(P_wtg)) * np.sinh(p_v_d));

    # Collect the differentiated symbolic output equations into a matrix
    dhdx_d = np.array([[1/v_wind, 0, 0       , 0],
                       [dhdv    , 0, dhdP_wtg, 0]])

    # Symbolic differentiated measurement function for the Kalman filter in the forecast method
    d_dhdx = ca.Function('d_dhdv', [y_d], [dhdx_d]);

    # Total measurement system for the forecast method
    y1 = np.log(v_wind);
    y2 = logit((0.5 + 0.5 * np.tanh(gamma1_d*(v_wind**2/4-gamma2_d)))*0.5*(1 + np.sin(P_wtg)));

    # Collect the symbolic output equations into a vector
    h_d = np.array([[y1], [y2]]);

    # Symbolic measurement function for the Kalman filter in the forecast method
    d_h = ca.Function('d_h', [y_d], [h_d]);

    return c_f, c_h, c_L,d_dhdx, d_h, system_info, system_var

''' Auxiliary function for battery system '''
def system_bat(x, u):
    """ Get battery system variables
        # Arguments:
            int/sym x     : battery system state Q (battery charge)
            int/sym u     : battery system input U (battery current) [A]

        # Outputs:
            int/sym P_bat : battery system power [MW]
            int/sym SOC   : battery state of charge [-]
            int/sym U_bat : battery voltage [V]
    """

    """ Battery constant parameters """
    # Fitting parameters of Shepherd's model
    E_bat_0 = 1.2848  ;  # Voltage constant [V]
    R = 0.002         ;  # Ohmic resistance [Ohm]
    K = 0.0091        ;  # Polarization constant [V]
    A = 0.111         ;  # Constant [V]
    B = 2.3077 / 3600 ;  # Constant [1/A s]
    Q = 6.5 * 3600    ;  # Maximum battery storage [A s]

    # Battery hyperparameters
    n_cell = 421*10*2  ;  # Number of cells per packages [-]
    n_packs = 626*4 ;  # Number of packages in the battery [-]

    """ Battery state of charge """
    SOC = (0.2 * Q - x) / Q; # [-]

    """ Battery electrochemistry """
    U_Ohm = R * u                           ; # Ohmic voltage [V]
    U_Pol = K / SOC                         ; # Polarization voltage [V]
    U_Exp = A * np.exp(-B * Q * (1 - SOC))  ; # Exponential voltage [V]
    U_OCV = E_bat_0 - U_Pol + U_Exp         ; # Open circuit voltage [V]
    U_bat = U_OCV - U_Ohm                   ; # Battery voltage [V]

    """ Battery power """
    P_bat = U_bat * u * n_packs * n_cell / (1000*1000);  # Battery net power [MW]

    return P_bat, SOC, U_bat

''' Auxiliary function for gas turbine system'''
def system_gtg(x):
    """ Get gas turbine system variables
        # Arguments:
            int/sym x          : gas turbine system state P (MW)

        # Outputs:
            int/sym P_gas      : modified gas turbine system power P [MW]
    """

    gtg_eta = 1; # Gas turbine generator efficiency

    """ Gas turbine power """
    P_gtg = x*gtg_eta;  # Gas turbine generator net power [MW]

    return P_gtg

''' Auxiliary function for the wind turbine system'''
def system_wtg(x, u):
    """ Get wind turbine generator system variables
        # Arguments:
            int/sym x          : wind turbine generator power output [MW]
            int/sym u          : wind turbine generator input P_wtg_curtailment [MW]

        # Outputs:
            int/sym P_wtg      : net wind turbine generator power [MW]
    """

    """ Wind turbine generator power """
    P_wtg = x-u; # Wind turbine generator net power [MW]

    return P_wtg

''' Auxiliary function for wind power system'''
def system_wind_curve(x,scale):
    """ Get wind power based on average wind speed using a regressed curve from data
        # Arguments:
            int x           : average wind speed [m/s^2]
        # Outputs:
            int P_wtg_pred      : predicted wind power output [MW]
    """

    P_wtg_pred = (-0.0089 + 0.0111*(x) - 0.0076*(x**2) + 0.0028*(x**3) - 0.0002*(x**4) + 0.000005*(x**5))*scale;
    return P_wtg_pred

''' Auxiliary function for logit '''
def logit(x):
    """ Get the inverse of the standard logistic function of a variable (theta(x)=1/(1+e^(-x)))
            # Arguments:
                sym/int x       : the variable to be used for the inverse of the standard logistic function
            # Outputs:
                sym/int x_logit : the variable after the inverse of the standard logistic function
        """

    x_logit = np.log(x/(1-x))
    return x_logit

''' Auxiliary function for deriving slacked chebyshev bounds '''
def cheby(mean,std,epsilon):
    beta = (1-epsilon)/(epsilon);
    return mean-(np.sqrt(beta)*std)/3

''' System class '''
class system_class:
    def __init__(self, system_info):
        """ Initialize dynamic model
        # Arguments:
            system_info      :   system_info : a dictionary with all of the relevant system information
                                 such as system dimensions, sampling time, process and measurement
                                 disturbance standard deviation, initial values ...
        """

        """ Initialisation """
        self._system_info = system_info; # Save system information in a dictionary
        self._wind_data = self.get_wind_data(savefig=False,plotfig=False); # Save wind data information in a dictionary

    def get_wind_data(self,savefig=False,plotfig=False):
        """ Plotting the average wind and power forecast for the case study
            # Input (optional):
                savefig         : a boolean to indicate whether to save the plots as a png or not
                plotfig         : a boolean to indicate whether to show the plots or not
            # Outputs:
                wind_data       : a dictionary with information about the wind, and the resulting
                                  wind power
        """

        """ Read data from csv files """
        dat5m = pd.read_csv(os.path.join(os.getcwd() + '/data/wind/dat_5m.csv')).to_numpy()[:, 1:];
        wind_perfect   = dat5m[:,1]; # Observed average wind speed [m/s^2]
        wind_imperfect = dat5m[:,2]; # Meteorlogical forecast of wind speed [m/s^2]
        wind_pred       = dat5m[:,3]; #
        P_wind_perfect = dat5m[:,4]; # Observed wind power output [pu]

        """ Collect data inside a dictionary """
        wind_data = {
            'wind_perfect': wind_perfect, 'wind_imperfect': wind_imperfect, # Average wind information
            'wind_pred':wind_pred,'P_wind_perfect' : P_wind_perfect, # Wind power information
            'wind_all':dat5m
        };

        """ Visualize data """
        if plotfig:
            ''' Plots '''
            plt.rcParams["figure.figsize"] = (16, 7.5);
            plt.rcParams["font.family"] = "Times New Roman";
            plt.rcParams['pdf.fonttype'] = 42;
            plt.rcParams['ps.fonttype'] = 42;

            # Create a 2x1 axes for plotting
            fig, axs = plt.subplots(2);

            # Create a time axis for plotting
            t = np.arange(0, len(wind_perfect)) / (12 * 24);

            # Compute the meteorological forecast based on inaccurate wind forecasts
            P_wind_met = system_wind_curve(wind_imperfect,1);

            # Clip meteorological forecasts as the wind curve is approximated with a polynomial
            P_wind_met[P_wind_met < 0] = 0; P_wind_met[P_wind_met > 1] = 1;

            # Plot the open loop predictions
            axs[0].plot(t, P_wind_met, linewidth=2, alpha=0.7,color=config_color_map['RWTH_red_100'], linestyle='--', label='meteorological forecast');
            axs[0].plot(t, P_wind_perfect, linewidth=2, alpha=0.7, color=config_color_map['NTNU_blue'], linestyle='--',label='observations [5 min]');
            axs[1].plot(t, wind_imperfect, alpha=0.7, linewidth=2, color=config_color_map['RWTH_red_100'],label='meteorological forecast');
            axs[1].plot(t, wind_perfect, linewidth=2, color=config_color_map['NTNU_blue'], label='observations [5 min]')

            # Auxiliary plotting configurations
            axs[0].set_ylabel('$P_\mathregular{wtg}$ [MW]',fontsize=15);
            axs[1].set_ylabel('$\mathregular{v}_\mathregular{wind}\mathregular{[m/v^2]}$',fontsize=15);
            axs[1].set_xlabel('$t$ [h]',fontsize=15);

            for i in range(2):
                axs[i].grid(True); axs[i].legend(fontsize=14); axs[i].set_xlim([0,408]);
                axs[i].tick_params(axis='x', labelsize=14)
                axs[i].tick_params(axis='y', labelsize=14)

            plt.tight_layout();

            if (savefig): fig.savefig('plot0_wind_data' + '.pdf', format='pdf', dpi=1200);

        return wind_data

    def simulate_empc_1_day(self,system_info,controller,n=1200,forecast_accuracy='perfect'):
        """ Plotting closed loop dynamics of the economic model predictive controller for 1 day = 24 hours
          # Input:
              system_info       : a dictionary with all of the relevant system information
                                  such as system dimensions, sampling time, process and measurment
                                  disturbance standard deviation, initial values ...
              controller        : a dynamic method instance for controlling the dynamical system (default: None)
              forecast_accuracy : a string which indicate whether the empc utilises observed, or meteorological
                                  forecasts ['perfect','imperfect','estimated']
          # Input (Optional):
              n                 : amount of open loop simulations (default: 1200)
        """
        print("***************************************************************"
              "******************************************");
        print("***************************************************************"
              "******************************************");
        print("Simulating the economic model predictive controller scheme for 1 day with "+forecast_accuracy+' forecasts')
        print("***************************************************************"
              "******************************************");

        """ Initial initialization """
        M = system_info['t_start'] * system_info['dt_v'];  # Data time frame for 1 day case study
        N = 24 * system_info['dt_v'];  # Prediction horizon corresponding to 1 day = 24 hours
        x_c = system_info['x0_c'];  # Initial system state

        # Initialization of meteorlogical forecast and actual observations of wind speed and power
        wind_real = self._wind_data['wind_perfect'][M:M + N];  # Observed wind speed for 1 day case study
        wind_met = self._wind_data['wind_imperfect'][M:M + N];  # Meteorological forecasts on wind speed for 1 day case
        P_wind_real = self._wind_data['P_wind_perfect'][M:M + N]*system_info['u_c_scale'][2];  # Wind power output for the 1 day case study
        P_wind_met = system_wind_curve(self._wind_data['wind_imperfect'][M:M + N],system_info['u_c_scale'][2]);  # Meteorological forecasts on wind power for 1 day case
        P_wind_met[P_wind_met < 0] = 0; # Clip meteorological forecasts
        P_wind_met[P_wind_met > system_info['u_c_scale'][2]] = system_info['u_c_scale'][2];  # Clip meteorological forecasts

        # Generate time grids for interpolation between observed data and the controller time step
        t_inter=np.arange(0.0,0.0+n*system_info['dt'],system_info['dt']); # Time grid for final interpolated wind power
        t_5s=np.arange(0.0,0.0+n*system_info['dt'],300); # Time grid for interpolation of sampled wind power

        """ Disturbance forecast initialization """
        P0_d_kf_L = system_info['P0_kf_d'];  # Initial system covariance for the disturbance model

        # Compute initial forecast
        _, P0_d_kf_L, x_d_mean_L,_ ,_ = controller.get_d_forecast_predictions(M, N, P0_d_kf_L,self._wind_data);

        # Compute the estimates in the nominal domain
        wind_mean = controller.get_c_v(x_d_mean_L[:,0]); # Initial estimation of the average wind speed
        P_wind_mean = controller.get_c_P(x_d_mean_L[:,0],x_d_mean_L[:,2]); # Initial estimation of the wind power output


        """ Interpolate the wind speed and power to the sampling time of the controller """
        wind_mean = np.interp(t_inter, t_5s, wind_mean);
        wind_real = np.interp(t_inter, t_5s, wind_real);
        wind_met = np.interp(t_inter, t_5s, wind_met);

        P_wind_mean = np.interp(t_inter, t_5s, P_wind_mean);
        P_wind_real = np.interp(t_inter, t_5s, P_wind_real);
        P_wind_met = np.interp(t_inter, t_5s, P_wind_met);

        """ Initialize the power demand """
        # Compute the power demand given maximum gas turbine power output and the meteorological forecasts
        P_demand = P_wind_met + system_info['x_c_scale'][1] * system_info['variable_power_scale'];

        """ Create sublists for the high level controller"""
        p_matrix = list(config_sublist_generator(list(P_wind_met), system_info['horizon_c']));
        r_matrix = list(config_sublist_generator(list(P_demand), system_info['horizon_c']));
        p_matrix_real = list(config_sublist_generator(list(P_wind_real), system_info['horizon_c']));


        ''' Initialization of vectors for plotting the cutted high level references '''
        x1_opt_list = np.zeros(n); x2_opt_list = np.zeros(n); x3_opt_list = np.zeros(n);
        u1_opt_list = np.zeros(n); u2_opt_list = np.zeros(n); u3_opt_list = np.zeros(n);
        P_wind_estimate = np.zeros(n); P_wind_estimates_all = np.zeros((len(t_5s) + 1, n));
        wind_estimate = np.zeros(n); wind_estimates_all = np.zeros((len(t_5s) + 1, n));

        ''' Initialization before simulation '''
        # Save the initial estimates from the sode before simulation
        wind_estimates_all[0, :] = wind_mean; P_wind_estimates_all[0, :] = P_wind_mean;

        # Set current wind power output estimate
        P_wind_mean_curr = P_wind_mean;

        # Initialize looping variables
        j = 0;  # Looping variable used to save each of the estimates from the sode
        k = 0;  # Looping variable used to keep track of estimates when they are not updated

        """ Start the simulation """
        with alive_bar(n) as bar:  # declare your expected total
            for i in range(n):

                # Modify the forecast such that they are perfectly known during the certainty horizon
                p_matrix[i][0:system_info['c_horizon_c']] = p_matrix_real[i][0:system_info['c_horizon_c']];

                """ Disturbance forecasting """
                # Sample and estimate wind speed and power using the disturbance forecast
                if(i % system_info['dt_d'] == 0):
                    # Update looping variables
                    k = 0; M += 1; j += 1;

                    # Compute the forecast estimates of wind speed and power
                    _,P0_d_kf_L, x_d_mean_L,_,_ = controller.get_d_forecast_predictions(M, N, P0_d_kf_L,self._wind_data)

                    # Compute the wind power output in the real world
                    P_wind_mean_curr = controller.get_c_P(x_d_mean_L[:,0],x_d_mean_L[:,2]);
                    v_wind_mean_curr = controller.get_c_v(x_d_mean_L[:,0]);

                    # Interpolate the wind speed and power to the sampling time of the controller
                    P_wind_mean_curr = np.interp(t_inter, t_5s, P_wind_mean_curr); # Interpolated wind power output estimates
                    wind_mean_curr =  np.interp(t_inter, t_5s, v_wind_mean_curr); # Interpolated wind speed estimates

                    # Save the forecast estimates of wind speed and power
                    P_wind_estimates_all[j,:] = P_wind_mean_curr
                    wind_estimates_all[j,:] =  wind_mean_curr

                # Save the wind speed and power forecast value at current time step at i
                P_wind_estimate[i] = P_wind_mean_curr[k]; wind_estimate[i] = wind_mean_curr[k];

                # Update the looping variable used to keep track of estimates when they are not updated
                k+=1;

                # Depending on forecast_accuracy, choose the controller forecast for wind power
                if (forecast_accuracy == 'perfect'):  # Given perfect knowledge of the observed values and their predictions
                    p_controller = p_matrix_real[i];
                elif (forecast_accuracy == 'imperfect'):  # Given imperfect knowledge of only meteorological forecasts
                    p_controller = p_matrix[i];
                elif (forecast_accuracy == 'estimate'):  # Given imperfect estimates from the disturbance forecasting
                    p_matrix_real[i][system_info['c_horizon_c']:] = \
                        P_wind_mean_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']];
                    p_controller = p_matrix_real[i];

                """ Controller """
                # Compute the references from the empc controller
                u_c, x_c,_ , solve_status_c,_ =  controller.get_c_empc_optimal_references(x_c,p_controller,r_matrix[i]);


                # Save optimal references from the empc controller
                u1_opt_list[i] = u_c[0]; u2_opt_list[i] = u_c[1]; u3_opt_list[i] = u_c[2];
                x1_opt_list[i] = x_c[0]; x2_opt_list[i] = x_c[1]; x3_opt_list[i] = x_c[2];

                """ Plant """
                # Get the open loop system states at the next recomputation point
                x_c = [x1_opt_list[i], x2_opt_list[i], x3_opt_list[i]];


                # Check controller solver status
                if (solve_status_c == 'Solve_Succeeded'):
                    warning = 1
                else:
                    print(i, 'EMPC_status:', solve_status_c);

                # Call after consuming one item for visual bar
                bar()

        """ Calculate the resulting performance of the control strategy """
        # Compute the scaled gas, wind, and battery power predictions
        P_gtg_curr = x2_opt_list * system_info['x_c_scale'][1];
        P_wtg_curr = P_wind_real[0:n] - u3_opt_list * system_info['u_c_scale'][2];
        P_bat_curr, SOC_bat_curr, U_bat_curr = list(
            zip(*map(system_bat,x3_opt_list*system_info['x_c_scale'][2],u2_opt_list*system_info['u_c_scale'][1])));

        scaling = (system_info['dt']/3600);

        # Compute the amount of power not satisfied (P_error), the total wind loss, and the total gas usage, forecast error
        print('P_error [MWH]:', sum(P_demand[0:n] - (P_gtg_curr + P_wtg_curr + P_bat_curr))*scaling);
        print('P_wtg_waste [MWH]:', sum(u3_opt_list * system_info['u_c_scale'][2])*scaling);
        print('P_gtg_sum [MWH]:', sum(P_gtg_curr)*scaling);
        print('(P_wtg_real - P_wind_met)_sum [MWH]:', sum(P_wind_real[0:n] - P_wind_met[0:n])*scaling);
        print('(P_wtg_real - P_wind_estimated)_sum [MWH]:', sum(P_wind_real[0:n] - P_wind_estimate[0:n])*scaling);
        print('abs(P_wtg_real - P_wind_met)_sum [MWH]:', sum(abs(P_wind_real[0:n] - P_wind_met[0:n]))*scaling);
        print('abs(P_wtg_real - P_wind_estimated)_sum [MWH]:', sum(abs(P_wind_real[0:n] - P_wind_estimate[0:n]))*scaling);


        # Save data points for later plotting
        with open('data/Saved/P_gtg_' + forecast_accuracy + '.pkl', 'wb') as f:
            pc.dump(P_gtg_curr, f)
        with open('data/Saved/P_wtg_' + forecast_accuracy + '.pkl', 'wb') as f:
            pc.dump(P_wtg_curr, f)
        with open('data/Saved/P_bat_' + forecast_accuracy + '.pkl', 'wb') as f:
            pc.dump(P_bat_curr, f)
        with open('data/Saved/SOC_bat_' + forecast_accuracy + '.pkl', 'wb') as f:
            pc.dump(SOC_bat_curr, f)

        with open('data/Saved/P_wind_mean_estimate.pkl', 'wb') as f:
            pc.dump(P_wind_estimate, f)
        with open('data/Saved/P_wind_mean_estimates_all.pkl', 'wb') as f:
            pc.dump(P_wind_estimates_all, f)
        with open('data/Saved/wind_mean_estimate.pkl', 'wb') as f:
            pc.dump(wind_estimate, f)
        with open('data/Saved/wind_mean_estimates_all.pkl', 'wb') as f:
            pc.dump(wind_estimates_all, f)

    def simulate_sempc_1_day(self,system_info,controller,chance_constraint,epsilon,n=1200):
        """ Plotting closed loop dynamics of the stochastic economic model predictive controller for 1 day = 24 hours
          # Input:
              system_info     : a dictionary with all of the relevant system information
                                such as system dimensions, sampling time, process and measurment
                                disturbance standard deviation, initial values ...
              controller      : a dynamic method instance for controlling the dynamical system (default: None)
          # Input (Optional):
              n               : amount of open loop simulations (default: 1200)
        """

        print("***************************************************************"
              "******************************************");
        print("***************************************************************"
              "******************************************");
        print("Simulating the stochastic economic model predictive controller scheme for 1 day with estimated forecasts")
        print("Epsilon:",epsilon)
        print("***************************************************************"
              "******************************************");


        """ Initial initialization """
        M = system_info['t_start'] * system_info['dt_v'];  # Data time frame for 1 day case study
        N = 24 * system_info['dt_v'];  # Prediction horizon corresponding to 1 day = 24 hours
        x_c = system_info['x0_c'];  # Initial system state

        epsilon1 = epsilon[0]; epsilon2 = epsilon[1]; epsilon3 = epsilon[2];

        # Initialization of meteorlogical forecast and actual observations of wind speed and power
        wind_real = self._wind_data['wind_perfect'][M:M + N];  # Observed wind speed for 1 day case study
        wind_met = self._wind_data['wind_imperfect'][M:M + N];  # Meteorological forecasts on wind speed for 1 day case
        P_wind_real = self._wind_data['P_wind_perfect'][M:M + N] * system_info['u_c_scale'][2];  # Wind power output for the 1 day case study
        P_wind_met = system_wind_curve(self._wind_data['wind_imperfect'][M:M + N], system_info['u_c_scale'][2]);  # Meteorological forecasts on wind power for 1 day case
        P_wind_met[P_wind_met < 0] = 0;  # Clip meteorological forecasts
        P_wind_met[P_wind_met > system_info['u_c_scale'][2]] = system_info['u_c_scale'][2];  # Clip meteorological forecasts

        t_inter = np.arange(0.0, 0.0 + n * system_info['dt'],system_info['dt']);  # Generate a time grid for final interpolated wind power
        t_5s = np.arange(0.0, 0.0 + n * system_info['dt'],300);  # Generate a time grid for interpolation of sampled wind power

        """ Disturbance forecast initialization """
        P0_d_kf_L = system_info['P0_kf_d'];  # Initial system covariance for the disturbance model

        # Compute initial forecast
        _, P0_d_kf_L, x_d_mean_L, x_d_var_L, _= controller.get_d_forecast_predictions(M, N, P0_d_kf_L,self._wind_data);

        # Compute the estimates in the nominal domain
        wind_mean = controller.get_c_v(x_d_mean_L[:, 0]);  # Initial estimation of the average wind speed
        P_wind_mean = controller.get_c_P(x_d_mean_L[:, 0], x_d_mean_L[:, 2]); # Initial estimation of the wind power output

        wind_std = np.sqrt(controller.get_c_v(x_d_var_L[:,0,0]));  # Initial estimation of the average wind speed std
        P_wind_std = controller.get_c_P_error(x_d_mean_L[:,2]+np.sqrt(x_d_var_L[:,2,2]),x_d_mean_L[:,2]); # Initial estimation of the wind power std output

        # Compute the estimates in the Lamperti domain
        wind_mean_L = x_d_mean_L[:,0]; # Initial estimation of average wind speed in Lamperti domain
        P_wind_mean_L = x_d_mean_L[:,2]; # Initial estimation of average wind power output in Lamperti domain

        wind_std_L = np.sqrt(x_d_var_L[:,0,0]); # Initial estimation of average wind speed standard deviation in Lamperti domain
        P_wind_std_L = np.sqrt(x_d_var_L[:,2,2]); # Initial estimation of power output standard deviation in Lamperti domain

        # Compute initial estimation of the backoff for chance constraints
        P_wind_cc_L = ss.norm.ppf(epsilon1, loc=P_wind_mean_L, scale=P_wind_std_L)
        P_wind_cc = ss.norm.ppf(epsilon2, loc=P_wind_mean, scale=P_wind_std)
        P_wind_cc_cheby = cheby(epsilon=epsilon3, mean=P_wind_mean, std=P_wind_std)

        # Interpolate the wind speed, power, and their standard deviations to the sampling time of the controller
        wind_mean = np.interp(t_inter, t_5s, wind_mean); wind_real = np.interp(t_inter, t_5s, wind_real);
        wind_met = np.interp(t_inter, t_5s, wind_met); wind_mean_L = np.interp(t_inter, t_5s, wind_mean_L)

        P_wind_mean = np.interp(t_inter, t_5s, P_wind_mean); P_wind_real = np.interp(t_inter, t_5s, P_wind_real);
        P_wind_met = np.interp(t_inter, t_5s, P_wind_met); P_wind_mean_L = np.interp(t_inter, t_5s, P_wind_mean_L);
        wind_std = np.interp(t_inter, t_5s, wind_std); wind_std_L = np.interp(t_inter, t_5s, wind_std_L)
        P_wind_std = np.interp(t_inter,t_5s, P_wind_std); P_wind_std_L = np.interp(t_inter, t_5s, P_wind_std_L)
        P_wind_cc_L = np.interp(t_inter, t_5s, P_wind_cc_L); P_wind_cc = np.interp(t_inter, t_5s, P_wind_cc)
        P_wind_cc_cheby = np.interp(t_inter, t_5s, P_wind_cc_cheby);

        ''' Init initialization '''
        # Case study initialization
        P_demand = P_wind_met + system_info['x_c_scale'][1]*system_info['variable_power_scale'];

        """ Create sublists for the high level controller"""
        r_matrix = list(config_sublist_generator(list(P_demand), system_info['horizon_c']));
        p_matrix_cc = list(config_sublist_generator(list(np.zeros(n)), system_info['horizon_c']));
        p_matrix_cc_L = list(config_sublist_generator(list(np.zeros(n)), system_info['horizon_c']));
        p_matrix_cc_cheby = list(config_sublist_generator(list(np.zeros(n)), system_info['horizon_c']));
        p_matrix_real = list(config_sublist_generator(list(P_wind_real), system_info['horizon_c']));

        ''' Initialization of vectors for plotting the cutted high level references '''
        x1_opt_list = np.zeros(n); x2_opt_list = np.zeros(n); x3_opt_list = np.zeros(n);
        u1_opt_list = np.zeros(n); u2_opt_list = np.zeros(n); u3_opt_list = np.zeros(n);

        P_wind_estimate = np.zeros(n); P_wind_estimates_all = np.zeros((len(t_5s) + 1, n));
        wind_estimate = np.zeros(n); wind_estimates_all = np.zeros((len(t_5s) + 1, n));

        P_wind_std_estimate = np.zeros(n); P_wind_std_estimates_all = np.zeros((len(t_5s) + 1, n));
        wind_std_estimate = np.zeros(n); wind_std_estimates_all = np.zeros((len(t_5s) + 1, n));

        P_wind_L_estimate = np.zeros(n); P_wind_L_estimates_all = np.zeros((len(t_5s) + 1, n));
        wind_L_estimate = np.zeros(n); wind_L_estimates_all = np.zeros((len(t_5s) + 1, n));

        P_wind_L_std_estimate = np.zeros(n); P_wind_L_std_estimates_all = np.zeros((len(t_5s) + 1, n));
        wind_L_std_estimate = np.zeros(n); wind_L_std_estimates_all = np.zeros((len(t_5s) + 1, n));

        P_wind_cc_cheby_estimate = np.zeros(n); P_wind_cc_cheby_estimates_all = np.zeros((len(t_5s) + 1, n));
        P_wind_cc_L_estimate = np.zeros(n); P_wind_cc_L_estimates_all = np.zeros((len(t_5s) + 1, n));
        P_wind_cc_estimate = np.zeros(n); P_wind_cc_estimates_all = np.zeros((len(t_5s) + 1, n));


        ''' Initialization before simulation '''
        # Save the initial estimates from the sode before simulation
        wind_estimates_all[0, :] = wind_mean; P_wind_estimates_all[0, :] = P_wind_mean;
        wind_std_estimates_all[0, :] = wind_std; P_wind_std_estimates_all[0, :] = P_wind_std;
        wind_L_estimates_all[0, :] = wind_mean_L; P_wind_L_estimates_all[0, :] = P_wind_mean_L;
        wind_L_std_estimates_all[0, :] = wind_std_L; P_wind_L_std_estimates_all[0, :] = P_wind_std_L;
        P_wind_cc_L_estimates_all[0, :] = P_wind_cc_L; P_wind_cc_estimates_all[0, :] = P_wind_cc;
        P_wind_cc_cheby_estimates_all[0,:] = P_wind_cc_cheby;


        # Set current wind power output estimate
        P_wind_mean_curr = P_wind_mean; P_wind_std_curr = P_wind_std; P_wind_cc_L_curr = P_wind_cc_L
        P_wind_mean_L_curr = P_wind_mean_L; P_wind_std_L_curr = P_wind_std_L; P_wind_cc_curr = P_wind_cc
        P_wind_cc_cheby_curr = P_wind_cc_cheby;

        # Initialize looping variables
        j = 0;  # Looping variable used to save each of the estimates from the sode
        k = 0;  # Looping variable used to keep track of estimates when they are not updated

        """ Start the simulation """
        with alive_bar(n) as bar:  # declare your expected total
            for i in range(n):
                """ Disturbance forecasting """
                # Sample and estimate wind speed and power using the disturbance forecast
                if (i % system_info['dt_d'] == 0):
                    # Update looping variables
                    k = 0; M += 1; j += 1;

                    # Compute the forecast estimates of wind speed and power
                    _, P0_d_kf_L, x_d_mean_L, x_d_var_L,_ = controller.get_d_forecast_predictions(M, N, P0_d_kf_L,self._wind_data)

                    # Compute the estimates and uncertainties in the real world
                    P_wind_mean_curr = controller.get_c_P(x_d_mean_L[:, 0], x_d_mean_L[:, 2]);
                    wind_mean_curr = controller.get_c_v(x_d_mean_L[:,0]);

                    P_wind_std_curr = controller.get_c_P_error(x_d_mean_L[:,2]+np.sqrt(x_d_var_L[:,2,2]),x_d_mean_L[:,2]);
                    wind_std_curr = np.sqrt(controller.get_c_v(x_d_var_L[:,0,0]));

                    # Compute the estimates and uncertainties in the Lamperti domain
                    P_wind_mean_L_curr = x_d_mean_L[:,2]; wind_mean_L_curr = x_d_mean_L[:,0];
                    P_wind_std_L_curr = np.sqrt(x_d_var_L[:,2,2]); wind_std_L_curr = np.sqrt(x_d_var_L[:,0,0]);

                    # Interpolate the wind speed and power to the sampling time of the controller
                    P_wind_mean_curr = np.interp(t_inter,t_5s,P_wind_mean_curr);  # Interpolated wind power output estimates
                    P_wind_std_curr = np.interp(t_inter,t_5s,P_wind_std_curr);  # Interpolated wind power std output estimates
                    wind_mean_curr = np.interp(t_inter,t_5s,wind_mean_curr);  # Interpolated wind speed estimates
                    wind_std_curr = np.interp(t_inter, t_5s,wind_std_curr);  # Interpolated wind speed std estimates

                    P_wind_mean_L_curr = np.interp(t_inter,t_5s,P_wind_mean_L_curr);  # Interpolated wind power std in Lamperti Domain
                    P_wind_std_L_curr =  np.interp(t_inter,t_5s,P_wind_std_L_curr);  # Interpolated wind power std in Lamperti Domain
                    wind_mean_L_curr = np.interp(t_inter, t_5s, wind_mean_L_curr);  # Interpolated wind speed in Lamperti Domain
                    wind_std_L_curr = np.interp(t_inter, t_5s, wind_std_L_curr);  # Interpolated wind speed std in Lamperti Domain

                    # Interpolated backoff constants
                    P_wind_cc_L_curr = ss.norm.ppf(epsilon1, loc=P_wind_mean_L_curr, scale=P_wind_std_L_curr)
                    P_wind_cc_curr = ss.norm.ppf(epsilon2, loc=P_wind_mean_curr, scale=P_wind_std_curr)
                    P_wind_cc_cheby_curr = cheby(epsilon=epsilon3, mean=P_wind_mean_curr, std=P_wind_std_curr)

                    # Save the forecast estimates of wind speed and power
                    P_wind_estimates_all[j, :] = P_wind_mean_curr; wind_estimates_all[j, :] = wind_mean_curr;
                    P_wind_std_estimates_all[j, :] = P_wind_std_curr; wind_std_estimates_all[j, :] = wind_std_curr;

                    P_wind_L_estimates_all[j, :] = P_wind_mean_L_curr; wind_L_estimates_all[j, :] = wind_mean_L_curr;
                    P_wind_L_std_estimates_all[j, :] = P_wind_std_L_curr; wind_std_estimates_all[j, :] = wind_std_L_curr;

                    P_wind_cc_L_estimates_all[j, :] = P_wind_cc_L_curr; P_wind_cc_estimates_all[j, :] = P_wind_cc_curr;
                    P_wind_cc_cheby_estimates_all[j, :] = P_wind_cc_cheby_curr;


                # Save the wind speed and power forecast value at current time step at i
                P_wind_estimate[i] = P_wind_mean_curr[k];    wind_estimate[i] = wind_mean_curr[k];
                P_wind_std_estimate[i] = P_wind_std_curr[k]; wind_std_estimate[i] = wind_std_curr[k];

                P_wind_L_estimate[i] = P_wind_mean_L_curr[k]; wind_L_estimate[i] = wind_mean_L_curr[k];
                P_wind_L_std_estimate[i] = P_wind_std_L_curr[k]; wind_L_std_estimate[i] = wind_std_L_curr[k];

                P_wind_cc_L_estimate[i] = P_wind_cc_L_curr[k]; P_wind_cc_estimate[i] = P_wind_cc_curr[k];
                P_wind_cc_cheby_estimate[i] = P_wind_cc_cheby_curr[k]

                # Update the looping variable used to keep track of estimates when they are not updated
                k += 1;

                # Given imperfect estimates from the disturbance forecasting
                p_matrix_real[i][system_info['c_horizon_c']:] = \
                    P_wind_mean_curr[k+system_info['c_horizon_c']:k + system_info['horizon_c']];
                p_controller = p_matrix_real[i];


                """ Controller """
                if(chance_constraint == 'lamperti'):
                    # Given imperfect uncertainty from the disturbance forecasting, compute the backoff term for chance constraints
                    cc_L_mean_curr = P_wind_mean_L_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    cc_L_std_curr = P_wind_std_L_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    p_matrix_cc_L[i][system_info['c_horizon_c']:] = ss.norm.ppf(epsilon1, loc=cc_L_mean_curr, scale=cc_L_std_curr)
                    p_cc_L_controller = p_matrix_cc_L[i];

                    # Compute the references from the s-empc controller
                    u_c, x_c, _, solve_status_c,_ = \
                        controller.get_c_sempc_optimal_references(x_c, p_controller, p_cc_L_controller, r_matrix[i]);

                elif(chance_constraint == 'gaussian'):
                    # Given imperfect uncertainty from the disturbance forecasting, compute the backoff term for chance constraints
                    cc_mean_curr = P_wind_mean_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    cc_std_curr = P_wind_std_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    p_matrix_cc[i][system_info['c_horizon_c']:] = ss.norm.ppf(epsilon2, loc=cc_mean_curr, scale=cc_std_curr)

                    p_cc_controller = np.array(p_matrix_cc[i]);
                    p_cc_controller[:system_info['c_horizon_c']] = p_controller[:system_info['c_horizon_c']];
                    p_cc_controller[p_cc_controller < 0] = 0;  # Clip meteorological forecasts
                    p_cc_controller[p_cc_controller > system_info['u_c_scale'][2]] = system_info['u_c_scale'][2];  # Clip meteorological forecasts

                    # Compute the references from the s-empc controller
                    u_c, x_c, _, solve_status_c,_ = controller.get_c_empc_optimal_references(x_c, p_cc_controller, r_matrix[i]);

                elif(chance_constraint == 'chebyshev'):
                    # Given imperfect uncertainty from the disturbance forecasting, compute the backoff term for chance constraints
                    cc_mean_curr = P_wind_mean_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    cc_std_curr = P_wind_std_curr[k + system_info['c_horizon_c']:k + system_info['horizon_c']]
                    p_matrix_cc_cheby[i][system_info['c_horizon_c']:] = cheby(epsilon=epsilon3, mean=cc_mean_curr, std=cc_std_curr)

                    p_cc_controller = np.array(p_matrix_cc_cheby[i]);
                    p_cc_controller[:system_info['c_horizon_c']] = p_controller[:system_info['c_horizon_c']];
                    p_cc_controller[p_cc_controller < 0] = 0;  # Clip meteorological forecasts
                    p_cc_controller[p_cc_controller > system_info['u_c_scale'][2]] = system_info['u_c_scale'][2];  # Clip meteorological forecasts

                    # Compute the references from the s-empc controller
                    u_c, x_c, _, solve_status_c,_ = controller.get_c_empc_optimal_references(x_c, p_cc_controller, r_matrix[i]);


                # Save optimal references from the empc controller
                u1_opt_list[i] = u_c[0]; u2_opt_list[i] = u_c[1]; u3_opt_list[i] = u_c[2];
                x1_opt_list[i] = x_c[0]; x2_opt_list[i] = x_c[1]; x3_opt_list[i] = x_c[2];

                """ Plant """
                # Get the open loop system states at the next recomputation point
                x_c = [x1_opt_list[i], x2_opt_list[i], x3_opt_list[i]]

                # Check high level controller solver status
                if (solve_status_c == 'Solve_Succeeded'):
                    warning = 1
                else:
                    print(i, 'SEMPC_status:', solve_status_c);

                bar()  # call after consuming one item for visual bar

        """ Calculate the resulting performance of the control strategy """
        # Compute the scaled gas, wind, and battery power predictions
        P_gtg_curr = x2_opt_list * system_info['x_c_scale'][1];
        P_wtg_curr = P_wind_real[0:n] - u3_opt_list * system_info['u_c_scale'][2];
        P_bat_curr, SOC_bat_curr, U_bat_curr = list(
            zip(*map(system_bat,x3_opt_list*system_info['x_c_scale'][2],u2_opt_list*system_info['u_c_scale'][1])));

        scaling = (system_info['dt']/3600);

        # Compute the amount of power not satisfied (P_error), the total wind loss, and the total gas usage, forecast error
        print('P_error [MWH]:', sum(P_demand[0:n] - (P_gtg_curr + P_wtg_curr + P_bat_curr))*scaling);
        print('P_wtg_waste [MWH]:', sum(u3_opt_list * system_info['u_c_scale'][2])*scaling);
        print('P_gtg_sum [MWH]:', sum(P_gtg_curr)*scaling);
        print('(P_wtg_real - P_wind_met)_sum [MWH]:', sum(P_wind_real[0:n] - P_wind_met[0:n])*scaling);
        print('(P_wtg_real - P_wind_estimated)_sum [MWH]:', sum(P_wind_real[0:n] - P_wind_estimate[0:n])*scaling);
        print('abs(P_wtg_real - P_wind_met)_sum [MWH]:', sum(abs(P_wind_real[0:n] - P_wind_met[0:n]))*scaling);
        print('abs(P_wtg_real - P_wind_estimated)_sum [MWH]:', sum(abs(P_wind_real[0:n] - P_wind_estimate[0:n]))*scaling);


        # Save data from applying the method on the system
        if(chance_constraint == 'lamperti'):
            # Save data points for later plotting
            with open('data/Saved/P_gtg_estimate_cc_' + chance_constraint + '_' + str(epsilon1) + '.pkl','wb') as f:
                pc.dump(P_gtg_curr, f)
            with open('data/Saved/P_wtg_estimate_cc_' + chance_constraint + '_' + str(epsilon1) + '.pkl','wb') as f:
                pc.dump(P_wtg_curr, f)
            with open('data/Saved/P_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon1) + '.pkl','wb') as f:
                pc.dump(P_bat_curr, f)
            with open('data/Saved/SOC_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon1) + '.pkl','wb') as f:
                pc.dump(SOC_bat_curr, f)
        elif(chance_constraint == 'gaussian'):
            # Save data points for later plotting
            with open('data/Saved/P_gtg_estimate_cc_' + chance_constraint + '_' + str(epsilon2) + '.pkl','wb') as f:
                pc.dump(P_gtg_curr, f)
            with open('data/Saved/P_wtg_estimate_cc_' + chance_constraint + '_' + str(epsilon2) + '.pkl','wb') as f:
                pc.dump(P_wtg_curr, f)
            with open('data/Saved/P_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon2) + '.pkl','wb') as f:
                pc.dump(P_bat_curr, f)
            with open('data/Saved/SOC_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon2) + '.pkl','wb') as f:
                pc.dump(SOC_bat_curr, f)
        elif(chance_constraint == 'chebyshev'):
            # Save data points for later plotting
            with open('data/Saved/P_gtg_estimate_cc_' + chance_constraint + '_' + str(epsilon3) + '.pkl','wb') as f:
                pc.dump(P_gtg_curr, f)
            with open('data/Saved/P_wtg_estimate_cc_' + chance_constraint + '_' + str(epsilon3) + '.pkl','wb') as f:
                pc.dump(P_wtg_curr, f)
            with open('data/Saved/P_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon3) + '.pkl','wb') as f:
                pc.dump(P_bat_curr, f)
            with open('data/Saved/SOC_bat_estimate_cc_' + chance_constraint + '_' + str(epsilon3) + '.pkl','wb') as f:
                pc.dump(SOC_bat_curr, f)


        with open('data/Saved/P_wind_backoff_estimate_cc_L_'+str(epsilon1)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_L_estimate, f)
        with open('data/Saved/P_wind_backoff_estimates_cc_L_all_'+str(epsilon1)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_L_estimates_all, f)

        with open('data/Saved/P_wind_backoff_estimate_cc_'+str(epsilon2)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_estimate, f)
        with open('data/Saved/P_wind_backoff_estimates_cc_all_'+str(epsilon2)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_estimates_all, f)

        with open('data/Saved/P_wind_backoff_estimate_cc_cheby_'+str(epsilon3)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_cheby_estimate, f)
        with open('data/Saved/P_wind_backoff_estimates_cc_cheby_all_'+str(epsilon3)+'.pkl', 'wb') as f:
            pc.dump(P_wind_cc_cheby_estimates_all, f)




        with open('data/Saved/P_wind_estimate.pkl', 'wb') as f:
            pc.dump(P_wind_estimate, f)
        with open('data/Saved/P_wind_estimates_all.pkl', 'wb') as f:
            pc.dump(P_wind_estimates_all, f)

        with open('data/Saved/P_wind_estimate_L.pkl', 'wb') as f:
            pc.dump(P_wind_L_estimate, f)
        with open('data/Saved/P_wind_estimates_L_all.pkl', 'wb') as f:
            pc.dump(P_wind_L_estimates_all, f)

        with open('data/Saved/P_wind_std_estimate_L.pkl', 'wb') as f:
            pc.dump(P_wind_L_std_estimate, f)
        with open('data/Saved/P_wind_std_estimates_L_all.pkl', 'wb') as f:
            pc.dump(P_wind_L_std_estimates_all, f)

        with open('data/Saved/P_wind_std_estimate.pkl', 'wb') as f:
            pc.dump(P_wind_std_estimate, f)
        with open('data/Saved/P_wind_std_estimates_all.pkl', 'wb') as f:
            pc.dump(P_wind_std_estimates_all, f)

    def compare_results_1_day(self,controller,system_info,n):
        """ Comparing closed loop dynamics of the predictive controllers for 1 day = 24 hours
              # Input:
                  system_info     : a dictionary with all of the relevant system information
                                    such as system dimensions, sampling time, process and measurment
                                    disturbance standard deviation, initial values ...
                  controller      : a dynamic method instance for controlling the dynamical system (default: None)
              # Input (Optional):
                  n               : amount of open loop simulations (default: 1200)
        """

        print("***************************************************************"
              "******************************************");
        print("***************************************************************"
              "******************************************");
        print("Comparing the predictive controllers for 1 day with perfect/imperfect/estimated forecasts")
        print("***************************************************************"
              "******************************************");

        """ Initial initialization """
        M = system_info['t_start'] * system_info['dt_v'];  # Data time frame for 1 day case study
        N = 24 * system_info['dt_v'];  # Prediction horizon corresponding to 1 day = 24 hours

        # Initialization of meteorlogical forecast and actual observations of wind speed and power
        wind_real = self._wind_data['wind_perfect'][M:M + N];  # Observed wind speed for 1 day case study
        wind_met = self._wind_data['wind_imperfect'][M:M + N];  # Meteorological forecasts on wind speed for 1 day case
        P_wind_real = self._wind_data['P_wind_perfect'][M:M + N] * system_info['u_c_scale'][2];  # Wind power output for the 1 day case study
        P_wind_met = system_wind_curve(self._wind_data['wind_imperfect'][M:M + N], system_info['u_c_scale'][2]);  # Meteorological forecasts on wind power for 1 day case
        P_wind_met[P_wind_met < 0] = 0;  # Clip meteorological forecasts
        P_wind_met[P_wind_met > system_info['u_c_scale'][2]] = system_info['u_c_scale'][2];  # Clip meteorological forecasts

        # Generate time grids for interpolation between observed data and the controller time step
        t_inter = np.arange(0.0, 0.0 + n * system_info['dt'],system_info['dt']);  # Time grid for final interpolated wind power
        t_5s = np.arange(0.0, 0.0 + n * system_info['dt'], 300);  # Time grid for interpolation of sampled wind power

        """ Interpolate the wind speed and power to the sampling time of the controller """
        wind_real = np.interp(t_inter, t_5s, wind_real);
        wind_met = np.interp(t_inter, t_5s, wind_met);

        P_wind_real = np.interp(t_inter, t_5s, P_wind_real);
        P_wind_met = np.interp(t_inter, t_5s, P_wind_met);

        """ Initialize the power demand """
        # Compute the power demand given maximum gas turbine power output and the meteorological forecasts
        P_demand = P_wind_met + system_info['x_c_scale'][1] * system_info['variable_power_scale'];

        """ Extract data for 1 day case study using nominal model predictive control with perfect forecasts """
        with open('data/Saved/P_gtg_perfect.pkl', 'rb') as f:
            P_gtg_perfect = pc.load(f)
        with open('data/Saved/P_wtg_perfect.pkl', 'rb') as f:
            P_wtg_perfect = pc.load(f)
        with open('data/Saved/P_bat_perfect.pkl', 'rb') as f:
            P_bat_perfect = pc.load(f)
        with open('data/Saved/SOC_bat_perfect.pkl', 'rb') as f:
            SOC_bat_perfect = pc.load(f)

        """ Extract data for 1 day case study using nominal model predictive control with meteorological forecasts """
        with open('data/Saved/P_gtg_imperfect.pkl', 'rb') as f:
            P_gtg_imperfect = pc.load(f)
        with open('data/Saved/P_wtg_imperfect.pkl', 'rb') as f:
            P_wtg_imperfect = pc.load(f)
        with open('data/Saved/P_bat_imperfect.pkl', 'rb') as f:
            P_bat_imperfect = pc.load(f)
        with open('data/Saved/SOC_bat_imperfect.pkl', 'rb') as f:
            SOC_bat_imperfect = pc.load(f)


        """ Extract data for 1 day case study using nominal model predictive control with estimated forecasts """
        with open('data/Saved/P_gtg_estimate.pkl', 'rb') as f:
            P_gtg_estimate = pc.load(f)
        with open('data/Saved/P_wtg_estimate.pkl', 'rb') as f:
            P_wtg_estimate = pc.load(f)
        with open('data/Saved/P_bat_estimate.pkl', 'rb') as f:
            P_bat_estimate = pc.load(f)
        with open('data/Saved/SOC_bat_estimate.pkl', 'rb') as f:
            SOC_bat_estimate = pc.load(f)
        with open('data/Saved/P_wind_mean_estimate.pkl', 'rb') as f:
            P_wind_mean_estimate = pc.load(f)
        with open('data/Saved/P_wind_mean_estimates_all.pkl', 'rb') as f:
            P_wind_mean_estimates_all = pc.load(f)

        """ Extract data for 1 day case study using stochastic model predictive control with estimated forecasts """
        with open('data/Saved/P_gtg_estimate_cc_lamperti_0.075.pkl', 'rb') as f:
            P_gtg_estimate_cc_L = pc.load(f)
        with open('data/Saved/P_wtg_estimate_cc_lamperti_0.075.pkl', 'rb') as f:
            P_wtg_estimate_cc_L = pc.load(f)
        with open('data/Saved/P_bat_estimate_cc_lamperti_0.075.pkl', 'rb') as f:
            P_bat_estimate_cc_L = pc.load(f)
        with open('data/Saved/SOC_bat_estimate_cc_lamperti_0.075.pkl', 'rb') as f:
            SOC_bat_estimate_cc_L = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimate_cc_L_0.075.pkl', 'rb') as f:
            P_wind_cc_L_estimate = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimates_cc_L_all_0.075.pkl', 'rb') as f:
            P_wind_cc_L_estimates_all = pc.load(f)

        """ Extract data for 1 day case study using conservative stochastic model predictive control with estimated forecasts """
        with open('data/Saved/P_gtg_estimate_cc_gaussian_0.2898.pkl', 'rb') as f:
            P_gtg_estimate_cc = pc.load(f)
        with open('data/Saved/P_wtg_estimate_cc_gaussian_0.2898.pkl', 'rb') as f:
            P_wtg_estimate_cc = pc.load(f)
        with open('data/Saved/P_bat_estimate_cc_gaussian_0.2898.pkl', 'rb') as f:
            P_bat_estimate_cc = pc.load(f)
        with open('data/Saved/SOC_bat_estimate_cc_gaussian_0.2898.pkl', 'rb') as f:
            SOC_bat_estimate_cc = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimate_cc_0.2898.pkl', 'rb') as f:
            P_wind_cc_estimate = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimates_cc_all_0.2898.pkl', 'rb') as f:
            P_wind_cc_estimates_all = pc.load(f)

        """ Extract data for 1 day case study using conservative stochastic model predictive control with estimated forecasts """
        with open('data/Saved/P_gtg_estimate_cc_chebyshev_0.266.pkl', 'rb') as f:
            P_gtg_estimate_cc_cheby = pc.load(f)
        with open('data/Saved/P_wtg_estimate_cc_chebyshev_0.266.pkl', 'rb') as f:
            P_wtg_estimate_cc_cheby = pc.load(f)
        with open('data/Saved/P_bat_estimate_cc_chebyshev_0.266.pkl', 'rb') as f:
            P_bat_estimate_cc_cheby = pc.load(f)
        with open('data/Saved/SOC_bat_estimate_cc_chebyshev_0.266.pkl', 'rb') as f:
            SOC_bat_estimate_cc_cheby = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimate_cc_cheby_0.266.pkl', 'rb') as f:
            P_wind_cc_cheby_estimate = pc.load(f)
        with open('data/Saved/P_wind_backoff_estimates_cc_cheby_all_0.266.pkl', 'rb') as f:
            P_wind_cc_cheby_estimates_all = pc.load(f)


        with open('data/Saved/P_wind_std_estimate.pkl', 'rb') as f:
            P_wind_std_estimate = pc.load(f)
        with open('data/Saved/P_wind_std_estimates_all.pkl', 'rb') as f:
            P_wind_std_estimates_all = pc.load(f)



        scaling = system_info['dt']/3600

        print('sum(P_error_perfect) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_perfect+P_bat_perfect+P_wtg_perfect)))
        print('sum(P_error_estimate+cc_L) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_estimate_cc_L+P_bat_estimate_cc_L+P_wtg_estimate_cc_L)))
        print('sum(P_error_estimate+cc_cheby) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_estimate_cc_cheby+P_bat_estimate_cc_cheby+P_wtg_estimate_cc_cheby)))
        print('sum(P_error_estimate+cc) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_estimate_cc+P_bat_estimate_cc+P_wtg_estimate_cc)))
        print('sum(P_error_estimate) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_estimate+P_bat_estimate+P_wtg_estimate)))
        print('sum(P_error_imperfect) [MWH]',scaling*sum(P_demand[:n]-(P_gtg_imperfect+P_bat_imperfect+P_wtg_imperfect)))

        print('sum(P_gtg_perfect): [MWH]',sum(P_gtg_perfect)*scaling)
        print('sum(P_gtg_imperfect): [MWH]',sum(P_gtg_imperfect)*scaling)
        print('sum(P_gtg_estimate): [MWH]',sum(P_gtg_estimate)*scaling)
        print('sum(P_gtg_estimate_cc_L): [MWH]',sum(P_gtg_estimate_cc_L)*scaling)
        print('sum(P_gtg_estimate_cc): [GW]',sum(P_gtg_estimate_cc)*scaling)
        print('sum(P_gtg_estimate_cc_chebyshev): [GW]',sum(P_gtg_estimate_cc_cheby)*scaling)

        print('sum(P_gtg_perfect): [%]', ((sum(P_gtg_perfect))/sum(P_gtg_perfect)) * 100-100)
        print('sum(P_gtg_imperfect): [%]', ((sum(P_gtg_imperfect))/sum(P_gtg_perfect)) * 100- 100)
        print('sum(P_gtg_estimate): [%]', ((sum(P_gtg_estimate))/sum(P_gtg_perfect)) * 100 - 100)
        print('sum(P_gtg_estimate_cc_L): [%]', ((sum(P_gtg_estimate_cc_L))/sum(P_gtg_perfect)) * 100 -100)
        print('sum(P_gtg_estimate_cc): [%]', ((sum(P_gtg_estimate_cc))/sum(P_gtg_perfect)) * 100 -100)
        print('sum(P_gtg_estimate_cc_chebyshev): [%]', ((sum(P_gtg_estimate_cc_cheby))/sum(P_gtg_perfect)) * 100 -100)



        # Compute the back off in nominal domain
        P_wind_estimate_cc_Linv = controller.get_c_P_simple(P_wind_cc_L_estimate);

        ''' Plot the results from the 1 day case study using nominal vs stochastic approaches '''
        t = np.arange(0.0,0.0+n*system_info['dt']*2,system_info['dt'])/3600; # Generate a time grid for plotting
        plt.rcParams["figure.figsize"] = (13.9, 8.72);
        plt.rcParams["font.family"] = "Times New Roman";
        plt.rcParams['pdf.fonttype'] = 42;
        plt.rcParams['ps.fonttype'] = 42;

        plt.rcParams["figure.figsize"] = (13.9, 7.22);

        fig, axs = plt.subplots(2, 2);

        # Prelimenary plotting to get the correct labels and legends (these are repeated under)
        leg1 = axs[0, 1].plot(t[:n], P_gtg_perfect, linewidth=3, color=config_color_map['NTNU_blue']);
        leg2 = axs[0, 1].plot(t[:n], P_gtg_imperfect, linewidth=3, color=config_color_map['RWTH_red_100']);
        leg3 = axs[0, 1].plot(t[:n], P_gtg_estimate, linewidth=3, color='orange');
        leg4 = axs[0, 1].plot(t[:n], P_gtg_estimate_cc_L, linewidth=3, color=config_color_map['Good_green']);

        # Plot the results from applying nominal model predictive control using perfect forecasts
        axs[0, 0].plot(t[:n], P_gtg_perfect + P_bat_perfect + P_wtg_perfect, linewidth=3,
                       color=config_color_map['NTNU_blue']);
        axs[0, 1].plot(t[:n], P_gtg_perfect, linewidth=3, color=config_color_map['NTNU_blue']);
        axs[1, 1].plot(t[:n], SOC_bat_perfect, linewidth=3, color=config_color_map['NTNU_blue']);

        # Plot the results from applying nominal model predictive control using meteorological forecasts
        axs[0, 0].plot(t[:n], P_gtg_imperfect + P_bat_imperfect + P_wtg_imperfect, linewidth=3,
                       color=config_color_map['RWTH_red_100']);
        axs[0, 1].plot(t[:n], P_gtg_imperfect, linewidth=3, color=config_color_map['RWTH_red_100']);
        axs[1, 1].plot(t[:n], SOC_bat_imperfect, linewidth=3, color=config_color_map['RWTH_red_100']);

        # Plot the results from applying nominal model predictive control using estimated forecasts
        axs[0, 0].plot(t[:n], P_gtg_estimate + P_bat_estimate + P_wtg_estimate, linewidth=3, color='orange');
        axs[0, 1].plot(t[:n], P_gtg_estimate, linewidth=3, color='orange');
        axs[1, 1].plot(t[:n], SOC_bat_estimate, linewidth=3, color='orange');

        # Plot the results from applying stochastic model predictive control using estimated forecasts
        axs[0, 0].plot(t[:n], P_gtg_estimate_cc_L + P_bat_estimate_cc_L + P_wtg_estimate_cc_L, linewidth=3,
                       color=config_color_map['Good_green']);
        axs[0, 1].plot(t[:n], P_gtg_estimate_cc_L, linewidth=3, color=config_color_map['Good_green']);
        axs[1, 1].plot(t[:n], SOC_bat_estimate_cc_L, linewidth=3, color=config_color_map['Good_green']);

        # Plot the power demand and wind forecast for this 1 day case study
        axs[1, 0].plot(t[:n], P_wind_real, linewidth=3, color=config_color_map['NTNU_blue'], label='observed values');
        axs[1, 0].plot(t[:n], P_wind_met, linewidth=3, color=config_color_map['RWTH_red_100'],label='meteorological forecast');
        axs[1, 0].plot(t[:n], P_wind_mean_estimate[:n], linewidth=3, color= 'orange', label='mean probablistic forecast');
        axs[0, 0].plot(t[:n], P_demand[:n], linewidth=4, color='k', linestyle='dotted', label='demand');

        # Plot the forecast estimates at intermediate points (only every 2, otherwise it would be too crowded)
        for i in range(0, len(t_5s), 4):
            P_wtg_curr = P_wind_mean_estimates_all[i][:system_info['horizon_c']]
            P_wind_std_curr = P_wind_std_estimates_all[i][:system_info['horizon_c']]
            t_curr = t[(i * 2):((i + 1) * 2) + (system_info['horizon_c'] - 2)];
            axs[1, 0].plot(t_curr, P_wtg_curr, linewidth=1.5, linestyle='--', alpha=0.9, color='k');  # Wind power forecast

            P_wind_mean_p_std_curr = P_wtg_curr + P_wind_std_curr
            P_wind_mean_m_std_curr = P_wtg_curr - P_wind_std_curr

            P_wind_mean_m_std_curr[P_wind_mean_m_std_curr < 0] = 0;  # Clip meteorological forecasts
            P_wind_mean_p_std_curr[P_wind_mean_p_std_curr > system_info['u_c_scale'][2]] = system_info['u_c_scale'][
                2];  # Clip meteorological forecasts

            axs[1, 0].fill_between(t_curr, P_wind_mean_m_std_curr, P_wind_mean_p_std_curr, color='k', alpha=0.05)

        axs[1, 0].plot(t[:n], P_wind_real, linewidth=3, color=config_color_map['NTNU_blue']);

        # Set the y and x label of the simulation, including the gridding
        axs[0, 0].set_ylabel('$P_\mathregular{total}$ [MW]', fontsize=17);
        axs[1, 0].set_ylabel('$P_\mathregular{wtg}$ [MW]', fontsize=17);
        axs[1, 1].set_ylabel('$\mathregular{SOC}_\mathregular{bat}$ [%]', fontsize=17);
        axs[0, 1].set_ylabel('$P_\mathregular{gtg}$ [MW]', fontsize=17);
        axs[1, 0].set_xlabel('$t$ [h]', fontsize=17);
        axs[1, 1].set_xlabel('$t$ [h]', fontsize=17);

        # Auxiliary plotting configurations
        for i in range(2):
            for j in range(2):
                axs[i, j].grid(True);
                axs[i, j].ticklabel_format(useOffset=False);
                axs[i, j].set_xlim([0, n * system_info['dt'] / 3600]);
                axs[i, j].tick_params(axis='x', labelsize=15)
                axs[i, j].tick_params(axis='y', labelsize=15)

        axs[0, 0].set_ylim([min(P_demand[:]) * 0.9, max(P_demand[:]) * 1.1]);
        axs[0, 0].legend(loc='upper right', fontsize=13);  # Set up legends
        axs[1, 0].legend(loc='upper right', fontsize=13);  # Set up legends
        plt.tight_layout(pad=3.5);  # Tighten the resulting plots
        plt.ticklabel_format(style='plain');  # Prevent scientific notation.

        # Legends and labels at the top of the figure
        line_labels = ["NMPC-PF", "NMPC-MF", "NMPC-EF", "SNMPC-EF"];

        fig.legend([leg1, leg2, leg3, leg4], labels=line_labels, loc="upper center",
                   borderaxespad=0, bbox_to_anchor=(0.530, 1), ncol=4, fontsize=18)

        # Save the graph as a pdf with a resolution of 1200 dpi
        fig.savefig('plot2_comparison_1_day_case_study_focus.pdf', format='pdf', dpi=1200)


        plt.rcParams["figure.figsize"] = (13.9, 8.72);

        """ Plot the wind power forecasts more closely """
        fig, axs = plt.subplots(1);
        # Plot the power demand and wind forecast for this 1 day case study
        axs.plot(t[:n], P_wind_real[:n], linewidth=3,color=config_color_map['NTNU_blue']);
        axs.plot(t[:n], P_wind_met[:n], linewidth=3,color=config_color_map['RWTH_red_100']);
        axs.plot(t[:n], P_wind_mean_estimate[:n], linewidth=3, color='orange');
        axs.plot(t[:n], P_wind_estimate_cc_Linv[:n],linewidth=3,color=config_color_map['Good_green'])


        # Set the y and x label of the simulation, including the gridding
        axs.set_ylabel('$P_\mathregular{wtg}$ [MW]', fontsize=38);
        axs.set_xlabel('$t$ [h]', fontsize=38);

        # Plot the forecast estimates at intermediate points (only every 2, otherwise it would be too crowded)
        for i in range(0, len(t_5s),4):
            axs.plot(t[(i * 2):((i + 1) * 2) + (system_info['horizon_c'] - 2)],
                           P_wind_mean_estimates_all[i][:system_info['horizon_c']],
                           linewidth=1.5, linestyle='--', alpha=0.9,
                           color='k');  # Wind power forecast
            axs.plot(t[(i * 2):((i + 1) * 2) + (system_info['horizon_c'] - 2)],
                           controller.get_c_P_simple(P_wind_cc_L_estimates_all[i][:system_info['horizon_c']]),
                           linewidth=1.5, linestyle='--', alpha=0.9, color='darkseagreen');  # Estimated+cc wind power forecast

        axs.plot(t[:n], P_wind_real[:n], linewidth=3,color=config_color_map['NTNU_blue']);

        # Auxiliary plotting configurations
        axs.grid(True);
        axs.ticklabel_format(useOffset=False);
        axs.set_xlim([0, n * system_info['dt'] / 3600]);
        axs.tick_params(axis='x', labelsize=35)
        axs.tick_params(axis='y', labelsize=35)

        # axs.legend(loc='upper right', fontsize=27);  # Set up legends
        plt.tight_layout();  # Tighten the resulting plots
        plt.ticklabel_format(style='plain');  # Prevent scientific notation.

        # Save the graph as a pdf with a resolution of 1200 dpi
        fig.savefig('plot2_comparison_1_day_case_study_wind_power_Linv.pdf', format='pdf', dpi=1200)
