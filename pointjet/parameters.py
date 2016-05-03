"""Pointjet simulation parameters."""

import numpy as np


# Simulation parameters
Lx = 2 * np.pi
Ly = 2 * np.pi
beta = 2 * 2 * np.pi
kappa = 1 / 1.5625
nu = 1e-3

# Reference jet
# cz = - A * sin(y/2)**2 * tanh((y-pi)/δ)
ref_amp = 0.6 * 2 * np.pi
ref_width = 0.05

# Vertical velocity perturbation:
#  ψ' = A * cos(k*x)
#  v' = - A * k * sin(k*x) = - V * sin(k*x)
#  css = A**2 * cos(k*ξ) / 2
pert_k = 2
pert_amp = 1e-2 / pert_k

# Discretization parameters
Nx = 64
Ny = 64
dt = 1e-3
stop_sim_time = 10
stop_wall_time = np.inf
stop_iteration = np.inf
mesh = (4,6)

