"""
Tests for ebisim.simulation._radial_dist
"""
# pylint: disable=missing-function-docstring

# import pytest
import numpy as np

from ebisim.simulation._radial_dist import(
    tridiagonal_matrix_algorithm,
    fd_system_uniform_grid,
    fd_system_nonuniform_grid,
    radial_potential_uniform_grid,
    radial_potential_nonuniform_grid,
    boltzmann_radial_potential_onaxis_density,
    boltzmann_radial_potential_linear_density
)
from ebisim.physconst import Q_E, M_E, EPS_0, PI


N = 50000
E_KIN = 5000.
V_E = np.sqrt(2 * E_KIN * Q_E/M_E)
R_E = 0.0001
R_D = 0.005
I = 1.
RHO_0 = -I/(V_E * PI * R_E**2)
def analytical_solution(r):
    f = r < R_E
    nf = np.logical_not(f)
    phi = np.zeros_like(r)
    phi0 = I/(4*PI*EPS_0*V_E)
    phi[f] = phi0 * ((r[f]/R_E)**2 + 2*np.log(R_E/R_D)-1)
    phi[nf] = phi0 * 2*np.log(r[nf]/R_D)
    return phi


def test_tdma():
    "Solves an arbirary tridiagonal system with matrix inversion (numpy) and tdma and compares"
    l = np.arange(6, 11)
    d = np.arange(1, 6)
    u = np.arange(2, 7)
    M = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)
    b = np.arange(20, 25)

    _x = np.linalg.inv(M).dot(b)
    x = tridiagonal_matrix_algorithm(l, d, u, b)
    np.testing.assert_allclose(_x, x)

    xpy = tridiagonal_matrix_algorithm.py_func(l, d, u, b)
    np.testing.assert_equal(xpy, x)

def test_fd_system_uniform_grid():
    dr = 10
    r = np.arange(0, 50, dr)
    _d = np.array([-2, -2, -2, -2, 0])/dr**2
    _d[-1] = 1
    _l = np.array([0, 1-0.5, 1-0.5/2, 1-0.5/3, 0])/dr**2
    _u = np.array([2, 1+0.5, 1+0.5/2, 1+0.5/3, 0])/dr**2

    l, d, u = fd_system_uniform_grid(r)
    np.testing.assert_allclose(_l, l)
    np.testing.assert_allclose(_d, d)
    np.testing.assert_allclose(_u, u)

    lpy, dpy, upy = fd_system_uniform_grid.py_func(r)
    np.testing.assert_equal(lpy, l)
    np.testing.assert_equal(dpy, d)
    np.testing.assert_equal(upy, u)

def test_fd_system_nonuniform_grid():
    r = np.array([0, 10, 20, 30, 40])
    _l, _d, _u = fd_system_uniform_grid(r)

    l, d, u = fd_system_nonuniform_grid(r)
    np.testing.assert_allclose(_l, l)
    np.testing.assert_allclose(_d, d)
    np.testing.assert_allclose(_u, u)

    r = np.array([0, 1, 3, 6, 10])
    _u = np.array([2, 0.5, 0.177777777777778, 0.089285714285714, 0])
    _l = np.array([0, 0, 0.1, 0.063492063492064, 0])
    _d = np.array([-2, -0.5, -0.277777777777778, -0.152777777777778, 1])

    l, d, u = fd_system_nonuniform_grid(r)
    np.testing.assert_allclose(_l, l)
    np.testing.assert_allclose(_d, d)
    np.testing.assert_allclose(_u, u)

    lpy, dpy, upy = fd_system_nonuniform_grid.py_func(r)
    np.testing.assert_equal(lpy, l)
    np.testing.assert_equal(dpy, d)
    np.testing.assert_equal(upy, u)

def test_radial_potential_uniform_grid():
    r = np.linspace(0, R_D, N)
    rho = np.zeros(N)
    rho[r < R_E] = RHO_0

    phi = radial_potential_uniform_grid(r, rho)
    phipy = radial_potential_uniform_grid.py_func(r, rho)
    phi_a = analytical_solution(r)

    assert np.sum((1-phi[:-1]/phi_a[:-1])**2)/N < 1e-6
    np.testing.assert_allclose(phi, phi_a, rtol=1e-3)
    np.testing.assert_equal(phipy, phi)

def test_radial_potential_nonuniform_grid():
    r = np.geomspace(R_E/100, R_D, N)
    r[0] = 0
    r[-1] = R_D
    rho = np.zeros(N)
    rho[r < R_E] = RHO_0

    phi = radial_potential_nonuniform_grid(r, rho)
    phipy = radial_potential_nonuniform_grid.py_func(r, rho)
    phi_a = analytical_solution(r)

    assert np.sum((1-phi[:-1]/phi_a[:-1])**2)/N < 1e-6
    np.testing.assert_allclose(phi, phi_a, rtol=5.e-4)
    np.testing.assert_equal(phipy, phi)

def test_boltzmann_radial_potential_linear_density():
    r = np.geomspace(R_E/100, R_D, N)
    r[0] = 0
    r[-1] = R_D
    rho = np.zeros(N)
    rho[r < R_E] = RHO_0

    NLI = np.array([4.2095955e+09, 2.2095955e+09])
    KT = np.array([150, 50])
    Q = np.array((5, 3))

    phi, n, shape = boltzmann_radial_potential_linear_density(
        r, rho, NLI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis]
        )
    # test that python and numba give same result
    phipy, npy, shapepy = boltzmann_radial_potential_linear_density.py_func(
        r, rho, NLI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis]
        )
    np.testing.assert_allclose(phipy, phi, atol=0, rtol=1e-6)
    np.testing.assert_allclose(npy, n, atol=0, rtol=1e-6)
    np.testing.assert_allclose(shapepy, shape, atol=0, rtol=1e-6)


    # np.testing.assert_equal(phipy, phi)

def test_boltzmann_radial_potential_onaxis_density():
    r = np.geomspace(R_E/100, R_D, N)
    r[0] = 0
    rho = np.zeros(N)
    rho[r < R_E] = RHO_0

    NI = np.array([2.51722401e+16, 7.16550624e+16])
    KT = np.array([150, 50])
    Q = np.array((5, 3))

    phi, n, shape = boltzmann_radial_potential_onaxis_density(
        r, rho, NI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis]
        )
    # test that python and numba give same result
    phipy, npy, shapepy = boltzmann_radial_potential_onaxis_density.py_func(
        r, rho, NI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis]
        )
    np.testing.assert_allclose(phipy, phi)
    np.testing.assert_allclose(npy, n)
    np.testing.assert_allclose(shapepy, shape)

    np.testing.assert_equal(phipy, phi)
    np.testing.assert_equal(npy, n)
    np.testing.assert_equal(shapepy, shape)
