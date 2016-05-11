"""Pointjet simulation parameters."""

import numpy as np


# Simulation parameters
Lx = 2 * np.pi
Ly = np.pi
beta = 2 * 2 * np.pi
kappa = 1 / 1.5625
nu = 1e-3

# Reference jet
# cz = - A * sin(y/2)**2 * tanh((y-pi)/δ)
ref_amp = 0.6 * 2 * np.pi
ref_width = 0.05

# Locally correlated perturbations
# css = A * exp(-(x**2 + (2*sin((y1-y0)/2))**2/2)/δ**2)
pert_amp = 1e-3
pert_width = 0.1

# Discretization parameters
Nx = 128
Ny = 64
dt = 3e-3
stop_sim_time = 40
stop_wall_time = np.inf
stop_iteration = np.inf
mesh = (8, 16)

