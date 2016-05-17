# README #

Package: dss_beta_plane  
URL: https://bitbucket.org/kburns/dss_beta_plane  
Author: Keaton J. Burns

## Overview ##

This package implements a direct statistical simulation (DSS) of barotropic beta-plane dynamics under a zonal average using Dedalus.

The equation set is based on the form presented in Tobias & Marston 2013:
http://adsabs.harvard.edu/abs/2013PhRvL.110j4502T

## Implementation ##

The Dedalus implementation solves for the first and second streamfunction cumulants over a 3D domain `(x, y0, y1)`, with `y` in `[a,b]`, analogous to `(ξ, y, y')` in the reference notation.

One dimensional functions, such as the first cumulant, are stored in a 'diagonal' representation in `(y0, y1)`.  That is, a 1D function `c(z)` would be stored as a 3D Dedalus field `C` such that, 

`C(x, y0, y1) = c(y0+y1-a)`

In terms of Fourier coefficients,

`<kx, ky0, ky1 | C> = δ(kx, 0) * δ(ky0, ky1) * <ky0 | c(y0)>`

This representation allows `C` to be utilized as a 1D function of either `y0` or `y1` simply by interpolation at `y1=a` or `y0=a`, respectively.

An operator called `FourierDiagonal` implements spectral interpolation along the diagonal of a bivariate Fourier series, and is used to extract the local part of the second cumulant, which is then stored in the diagonal format described above. I.e. for `g = Diag(f)`,

`g(x, y0, y1) = f(x, y0+y1-a, y0+y1-a)`