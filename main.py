"""
# Main script for the collaboration project between Kiet Tuan Hoang (NTNU) and Christian
# Ankerstjerne Thilker. Main idea is to investigate advanced forecasting for offshore hybrid
# power systems consisting of gas turbine generators, offshore wind, and batteries. The batteries
# and the gas turbines are modelled using ordinary differential equations while wind is modelled
# as a disturbance in terms of stochastic nonlinear differential equations in the Lamperti domain.

# Copyright (c) 2021, Kiet Tuan Hoang/ Christian Ankerstjerne Thilker
# Last edited: 24.07.2022
"""

"""
Dependencies information (modules)
system  : a library for system specific functions
method  : a library for control and estimation specific functions

*system_class    : a class which acts as the simulator
*method_class    : a class which acts as the controller 
*system_OHPS     : a function which returns all of the system dynamics and variables
"""

''' Module imports '''
from method import method_class
from system import system_class
from system import system_OHPS

"""
Dependencies information (libraries)
* datetime : a library for time and space for documentation
* matplotlib     : a general library for plotting results 
"""


''' Auxillarily Imports '''
from datetime import datetime
import matplotlib.pyplot as plt

''' Abbreviations '''
# Offshore hybrid power system is abbreviated as ohps
# General model predictive control is abbreviated as mpc
# Economic model predictive control is abbreviated as empc
# Stochastic model predictive control is abbreviated as sempc
# Stochastic differential equation is abbreviated as sode
# Chance constraints are abbreviated as cc
# Lamperti-transformed functions have _L


''' Press the green button in the gutter to run the script '''
if __name__ == '__main__':
    print('*********************************************************************************************************')
    print('---------------------------------------------------------------------------------------------------------')
    print('Stochastic economic model predictive control based on')
    print('advanced forecasting of offshore hybrid power systems.')
    print('Current date and time at the start of code execution: ',datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print('---------------------------------------------------------------------------------------------------------')
    print('*********************************************************************************************************')

    """ Initializations """
    # Initial definitions
    n_day = 576; # Simulation study duration [t*dt] for 1 day = 24 hours
    n_week = n_day*7; # Simulation study duration [t*dt] for 1 week = 24*7 hours
    n_year = 105190*2;  # Simulation study duration [t*dt] for all of the data 1 year = 365.24 days

    # Get offshore hybrid power system dynamics and cost function
    c_f, c_h, c_L, d_dhdx,d_h, system_info, system_var = system_OHPS();
    epsilon_0    = [system_info['epsilon_cc_c'],system_info['epsilon_cc2_c'],system_info['epsilon_cc3_c']];
    epsilon_list = [round(0.025 + i * 0.025, 3) for i in range(int((0.45 - 0.025) / 0.025) + 1)];

    # Create a method_class instance describing the controller
    method = method_class(system_info=system_info, system_var=system_var,
                          c_f=c_f, c_h=c_h, c_L=c_L,d_dhdx=d_dhdx, d_h=d_h);

    # Create a system_class instance describing the offshore hybrid power system
    plant = system_class(system_info=system_info);


    """ Simulation for 1 day case study """
    plant.simulate_empc_1_day(system_info=system_info,controller=method,n=n_day,forecast_accuracy='perfect');
    plant.simulate_empc_1_day(system_info=system_info,controller=method,n=n_day,forecast_accuracy='imperfect');
    plant.simulate_empc_1_day(system_info=system_info,controller=method,n=n_day,forecast_accuracy='estimate');
    plant.simulate_sempc_1_day(system_info=system_info,controller=method,epsilon=epsilon_0, n=n_day,chance_constraint='lamperti');
    plant.simulate_sempc_1_day(system_info=system_info,controller=method,epsilon=epsilon_0, n=n_day,chance_constraint='gaussian');
    plant.simulate_sempc_1_day(system_info=system_info,controller=method,epsilon=epsilon_0, n=n_day,chance_constraint='chebyshev');
    plant.compare_results_1_day(system_info=system_info,controller=method,n=n_day) # Plotting results for 1 day

    plt.show()

    print('---------------------------------------------------------------------------------------------------------')
    print('Current date and time at the end of code execution: ',datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


