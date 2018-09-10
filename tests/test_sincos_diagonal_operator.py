"""SinCos diagonal operator tests."""

import numpy as np
from dedalus import public as de
from diagonal import SinCosDiagonal
de.operators.parseables['Diag'] = Diag = SinCosDiagonal

import logging
logger = logging.getLogger(__name__)


def general_test(test_func, py0, py1):

    # Parameters
    Cx = 0
    Cy = 0
    Lx = 2*np.pi
    Ly = np.pi
    Nx = 16
    Ny = 16

    # Bases and domain
    x_basis = de.Fourier('x', Nx, [Cx, Cx+Lx], dealias=3/2)
    y0_basis = de.SinCos('y0', Ny, [Cy, Cy+Ly], dealias=3/2)
    y1_basis = de.SinCos('y1', Ny, [Cy, Cy+Ly], dealias=3/2)
    domain = de.Domain([x_basis, y0_basis, y1_basis], grid_dtype=np.float64)

    # Test fields
    x, y0, y1 = domain.grids()
    a = Cy - Ly/2
    interp = de.operators.parseables['interp']

    f = domain.new_field()
    f.meta['y0']['parity'] = py0
    f.meta['y1']['parity'] = py1
    h = domain.new_field()
    h.meta['y0']['parity'] = 1
    h.meta['y1']['parity'] = py0*py1
    f['g'] = test_func(x, y0, y1)
    h['g'] = test_func(x, y1, y1)
    g = Diag(f, 'y0', 'y1').evaluate()

    print(g['c'])
    print()
    print(h['c'])
    return np.allclose(g['c'], h['c'])


def test_1d_n1():
    py0, py1 = -1, 1
    test_func = lambda x, y0, y1: np.sin(2*y0)
    assert general_test(test_func, py0, py1)

def test_1d_n2():
    py0, py1 = 1, 1
    test_func = lambda x, y0, y1: np.cos(2*y0)
    assert general_test(test_func, py0, py1)

def test_1d_n3():
    py0, py1 = 1, -1
    test_func = lambda x, y0, y1: np.sin(2*y1)
    assert general_test(test_func, py0, py1)

def test_1d_n4():
    py0, py1 = 1, 1
    test_func = lambda x, y0, y1: np.cos(2*y1)
    assert general_test(test_func, py0, py1)

def test_2d_n1():
    py0, py1 = 1, 1
    test_func = lambda x, y0, y1: np.cos(2*y0)*np.cos(3*y1)
    assert general_test(test_func, py0, py1)

def test_2d_n2():
    py0, py1 = -1, 1
    test_func = lambda x, y0, y1: np.sin(2*y0)*np.cos(3*y1)
    assert general_test(test_func, py0, py1)

def test_2d_n3():
    py0, py1 = 1, -1
    test_func = lambda x, y0, y1: np.cos(2*y0)*np.sin(3*y1)
    assert general_test(test_func, py0, py1)

def test_2d_n4():
    py0, py1 = -1, -1
    test_func = lambda x, y0, y1: np.sin(2*y0)*np.sin(3*y1)
    assert general_test(test_func, py0, py1)

def test_3d_n1():
    py0, py1 = 1, 1
    test_func = lambda x, y0, y1: np.cos(x)*np.cos(2*y0)*np.cos(2*y1)
    assert general_test(test_func, py0, py1)
