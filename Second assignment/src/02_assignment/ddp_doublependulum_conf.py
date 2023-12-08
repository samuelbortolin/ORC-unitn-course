# -*- coding: utf-8 -*-

import numpy as np
from math import pi, sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

q0 = np.array([ pi , 0.])    # initial configuration
qT = np.array([ 0. , 0])  # goal configuration
dt = 0.005                       # DDP time step
N = 400                         # horizon size

dt_sim = 1e-3                    # time step used for the final simulation
ndt = 1                          # number of integration steps for each control loop

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(2)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 10.0

SELECTION_MATRIX = 0            # flag to use the selection matrix method
ACTUATION_PENALTY = 1           # flag to use the actuation penalty method

use_viewer = True
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.2517311573028564, 0.6763767004013062, 0.28195011615753174, 0.3313407003879547, 0.557260274887085, 0.651939868927002, 0.3932540714740753]
