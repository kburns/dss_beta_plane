"""Pointjet simulation parameters."""

import numpy as np


# Simulation parameters
Lx = 2 * np.pi
Ly = 2 * np.pi
beta = 8
kappa = 1e-2
nu = 1e-3

# Reference jet
# cz = - A * sin(y/2)**2 * tanh((y-pi)/δ)
ref_amp = 2
ref_width = 0.2

# Vertical velocity perturbation:
#  ψ' = A * cos(k*x)
#  v' = - A * k * sin(k*x) = - V * sin(k*x)
#  css = A**2 * cos(k*ξ) / 2
pert_k = 2
pert_amp = 1e-2 / pert_k

# Discretization parameters
Nx = 32
Ny = 64
dt = 1e-2
stop_sim_time = 100
stop_wall_time = np.inf
stop_iteration = np.inf

