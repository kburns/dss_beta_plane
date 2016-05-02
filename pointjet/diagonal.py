"""Fourier diagonal operator definition."""

import numpy as np

from dedalus.core.field import Operand
from dedalus.core.operators import Operator, FutureField, Interpolate
from dedalus.tools.array import reshape_vector, axslice


class FourierDiagonal(Operator, FutureField):
    """
    Double-Fourier interpolation-on-diagonal operator.

    Parameters
    ----------
    arg : field object
        Field argument
    basis0, basis1 : basis identifiers
        Bases for diagonal interpolation

    Notes
    -----
    The return data is structured such that, for g = Diag(f),
        g(x1, x2) = f(x1+x2, x1+x2)
    s.t. interpolation of g at x1=0 or x2=0 yields the diagonalization of f,
    i.e. f(x, x), aranged as a one-dimensional function of x2 or x1, respectively.

    """

    def __init__(self, arg, basis0, basis1, **kw):
        arg = Operand.cast(arg)
        super().__init__(arg, **kw)
        self.basis0 = self.domain.get_basis_object(basis0)
        self.basis1 = self.domain.get_basis_object(basis1)
        self.axis0 = self.domain.bases.index(self.basis0)
        self.axis1 = self.domain.bases.index(self.basis1)
        if self.axis0 > self.axis1:
            raise ValueError("Cannot evaluate specified axis order.")
        self.name = 'Diag[%s=%s]' %(self.basis0.name, self.basis1.name)
        # Shear array
        k0 = reshape_vector(self.basis0.wavenumbers, dim=self.domain.dim, axis=self.axis0)
        x1 = self.domain.grid(self.axis1, scales=self.domain.dealias)
        self.shear = np.exp(1j*k0*x1)
        # Filter mask
        slices = self.domain.dist.coeff_layout.slices(self.domain.dealias)
        k0 = reshape_vector(self.basis0.wavenumbers[slices[self.axis0]], dim=self.domain.dim, axis=self.axis0)
        k1 = reshape_vector(self.basis1.wavenumbers[slices[self.axis1]], dim=self.domain.dim, axis=self.axis1)
        self.filter_mask = (k0 == k1)

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def check_conditions(self):
        # Shearing layout
        layout = self.args[0].layout
        return ((layout.grid_space[self.axis1]) and
                (not layout.grid_space[self.axis0]) and
                (layout.local[self.axis0]))

    def operate(self, out):
        arg = self.args[0]
        axis0 = self.axis0
        axis1 = self.axis1
        # Enforce conditions for shearing space
        arg.require_grid_space(axis=axis1)
        arg.require_coeff_space(axis=axis0)
        arg.require_local(axis=axis0)
        # Apply Fourier shear to flatten the diagonal
        # s.t out(y0, y1) = arg(y0+y1, y1)
        out.layout = arg.layout
        np.multiply(arg.data, self.shear, out=out.data)
        # Interpolate on flattened diagonal
        # s.t. out(y0, y1) = arg(y1, y1)
        self.basis0.Interpolate(out, 'left', out=out).evaluate()
        out.meta['y0']['constant'] = False
        # Broadcast and filter coefficients
        # s.t. out(y0, y1) = arg(y0+y1, y0+y1)
        out.data[axslice(axis0, None, None)] = out.data[axslice(axis0, 0, 1)]
        out.require_coeff_space(axis=axis1)
        out.data *= self.filter_mask
        # Move back to starting layout
        out.require_layout(arg.layout)

