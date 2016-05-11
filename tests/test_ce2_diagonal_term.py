"""CE2 diagonal term tests."""

import numpy as np
from dedalus import public as de
from dedalus.tools.array import reshape_vector
from diagonal import FourierDiagonal
de.operators.parseables['Diag'] = Diag = FourierDiagonal

import logging
logger = logging.getLogger(__name__)


def compute_second_cumulant(domain, f, g):
    """
    Compute second cumulant of two 2D functions under x' averaging:
        cfg(x, y0, y1) = <f(x'+x, y0) g(x', y1)>

    Parameters
    ----------
    domain : domain object
        Domain
    f, g: functions of (x, y)
        Fields over which to compute second cumulant

    Notes
    -----
    Assumes triple-Fourier real domain over (x, y0, y1).

    """
    x, y0, y1 = domain.grids()
    # Build 2D fields as functions of (x,y0) and (x,y1)
    a = domain.new_field()
    a['g'] = f(x, y0)
    b = domain.new_field()
    b['g'] = g(x, y1)
    # Compute cumulant directly from Fourier modes
    c = domain.new_field()
    kx = reshape_vector(domain.bases[0].wavenumbers, dim=3, axis=0)
    shift = np.exp(1j * kx * domain.bases[0].interval[0])
    c['c'] = a['c'][:,:,0:1] * np.roll(b['c'],-1,2)[:,0:1,::-1].conj() * shift
    return c


def general_test(test_func):

    # Parameters
    Cx = 4
    Cy = 7
    Lx = 2 * np.pi
    Ly = np.pi
    Nx = 32
    Ny = 32

    # Bases and domain
    x_basis = de.Fourier('x', Nx, [Cx-Lx/2, Cx+Lx/2], dealias=3/2)
    y0_basis = de.Fourier('y0', Ny, [Cy-Ly/2, Cy+Ly/2], dealias=3/2)
    y1_basis = de.Fourier('y1', Ny, [Cy-Ly/2, Cy+Ly/2], dealias=3/2)
    domain = de.Domain([x_basis, y0_basis, y1_basis], grid_dtype=np.float64)

    # Operators
    dx = x_basis.Differentiate
    dy0 = y0_basis.Differentiate
    dy1 = y1_basis.Differentiate
    interp = de.operators.parseables['interp']

    L0 = lambda A: dx(dx(A)) + dy0(dy0(A))
    L1 = lambda A: dx(dx(A)) + dy1(dy1(A))
    D = lambda A: Diag(interp(A, x=0), 'y0', 'y1')

    # Build cumulants
    css = compute_second_cumulant(domain, test_func, test_func)
    czs = L0(css).evaluate()
    csz = L1(css).evaluate()

    # Test different RHS term formulations
    form1 = D(dx(dy0(csz) + dy1(csz))).evaluate()
    form2 = D(dx(dy0(csz - czs))).evaluate()

    return np.allclose(form1['g'], form2['g'])


def test_1():
    test_func = lambda x, y: np.cos(2*x + 4*y) - np.cos(3*x - y) + np.sin(12*x)*np.sin(2*y)
    assert general_test(test_func)

