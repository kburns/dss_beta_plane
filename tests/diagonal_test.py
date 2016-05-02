"""Diag operator tests."""

import numpy as np
from dedalus import public as de

from diagonal import FourierDiagonal
de.operators.parseables['Diag'] = Diag = FourierDiagonal

import logging
logger = logging.getLogger(__name__)


def general_test(test_func):

    Lx = 2 * np.pi
    Ly = 2 * np.pi
    Nx = 16
    Ny = 16

    # Bases and domain
    x_basis = de.Fourier('x', Nx, [0, Lx], dealias=3/2)
    y0_basis = de.Fourier('y0', Ny, [0, Ly], dealias=3/2)
    y1_basis = de.Fourier('y1', Ny, [0, Ly], dealias=3/2)
    domain = de.Domain([x_basis, y0_basis, y1_basis], grid_dtype=np.float64)

    # Test fields
    x, y0, y1 = domain.grids()
    interp = de.operators.parseables['interp']

    f = domain.new_field()
    h = domain.new_field()
    f['g'] = test_func(x, y0, y1)
    h['g'] = test_func(0, y0+y1, y0+y1)
    g = Diag(interp(f, x=0), 'y0', 'y1').evaluate()

    return np.allclose(g['c'], h['c'])


def test_1d_y0():
    test_func = lambda x, y0, y1: np.cos(3*y0 + np.pi/3)
    assert general_test(test_func)

def test_1d_y1():
    test_func = lambda x, y0, y1: np.cos(3*y1 + np.pi/3)
    assert general_test(test_func)

def test_2d_n1():
    test_func = lambda x, y0, y1: np.cos(y0 + y1 + np.pi/4)
    assert general_test(test_func)

def test_2d_n2():
    test_func = lambda x, y0, y1: np.cos(2*y0 + y1 + np.pi/5)
    assert general_test(test_func)

def test_2d_n3():
    test_func = lambda x, y0, y1: np.cos(y0 + 2*y1 + np.pi/6)
    assert general_test(test_func)

def test_2d_n4():
    test_func = lambda x, y0, y1: np.cos(2*y0 + 2*y1 + np.pi/7)
    assert general_test(test_func)
