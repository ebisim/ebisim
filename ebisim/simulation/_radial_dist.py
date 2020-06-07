import numpy as np
from numba import njit

from..physconst import EPS_0, Q_E

@njit(cache=True)
def tdma(l, d, u, b):
    """
    Tridiagonal Matrix Algorithm [1]
    Solves a system of equations M x = b for x, where M is a tridiagonal matrix.

    Parameters
    ----------
    l : np.ndarray
        Lower diagonal vector l[i] = M[i, i-1].
    d : np.ndarray
        Diagonal vector d[i] = M[i, i].
    u : np.ndarray
        Upper diagonal vector u[i] = M[i, i+1].
    b : np.ndarray
        [description]

    Returns
    -------
    x : np.ndarray
        Solution.

    References
    ----------
    .. [1] "Tridiagonal matrix algorithm"
           https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = l.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    x  = np.zeros(n)
    cp[0] = u[0]/d[0]
    dp[0] = b[0]/d[0]
    for k in range(1, n):
        cp[k] = u[k]               /(d[k]-l[k]*cp[k-1])
        dp[k] = (b[k]-l[k]*dp[k-1])/(d[k]-l[k]*cp[k-1])
    x[-1] = dp[-1]
    for k in range(n-2, -1 , -1):
        x[k] = dp[k] - cp[k]*x[k+1]
    return x

@njit(cache=True)
def _td_uniform_grid(r):
    """
    Sets up the three diagonal vectors for a finite Poisson problem with radial symmetry on a
    uniform grid.
    d phi/dr = 0 at r = 0, and phi = phi0 at r = (n-1) * dr = r_max

    Parameters
    ----------
    r : np.ndarray
        <m>
        Grid node positions (are assumed to be evenly spaced)

    Returns
    -------
    l : np.ndarray
        Lower diagonal vector.
    d : np.ndarray
        Diagonal vector.
    u : np.ndarray
        Upper diagonal vector.
    """
    dr = r[1] - r[0]
    n = r.size

    d = np.full(n, -2/dr**2)
    d[-1] = 1

    l = np.zeros(n)
    l[1:-1] = (1-0.5/np.arange(1, n-1))/dr**2

    u = np.zeros(n)
    u[0] = 2/dr**2
    u[1:-1] = (1+0.5/np.arange(1, n-1))/dr**2

    return l, d, u

@njit(cache=True)
def radial_potential_uniform_grid(r, rho):
    """
    Solves the radial Poisson equation on a uniform grid.
    Boundary conditions are d phi/dr = 0 at r = 0 and phi(rmax) = 0

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, must be evenly spaced, with r[0] = 0, r[-1] = r_max.
    rho : np.ndarray
        <C/m^3>
        Charge density at r.

    Returns
    -------
    np.ndarray
        <V>
        Potential at r.
    """
    l, d, u = _td_uniform_grid(r)
    rho_ = rho[:]
    rho_[-1] = 0 #Boundary condition
    phi = tdma(l, d, u, -rho/EPS_0)
    return phi

@njit(cache=True)
def _td_nonuniform_grid(r):
    """
    Sets up the three diagonal vectors for a finite Poisson problem with radial symmetry on a
    nonuniform grid.
    d phi/dr = 0 at r = 0, and phi = phi0 at r = (n-1) * dr = r_max

    Parameters
    ----------
    r : np.ndarray
        <m>
        Grid node positions (are assumed to be evenly spaced)

    Returns
    -------
    l : np.ndarray
        Lower diagonal vector.
    d : np.ndarray
        Diagonal vector.
    u : np.ndarray
        Upper diagonal vector.
    """
    dr = r[1:] - r[:-1]
    n = r.size
    # print(dr)
    weight1 = 2/(dr[1:] * dr[:-1] * (dr[1:] + dr[:-1]))
    weight2 = 1/(r[1:-1]*(dr[1:]**2 * dr[:-1] + dr[1:] * dr[:-1]**2))
    # print(weight1)
    # print(weight2)
    d = np.zeros(n)
    d[0] = -2/dr[0]**2
    d[1:-1] = -(dr[:-1] + dr[1:]) * weight1 + (dr[1:]**2 - dr[:-1]**2) * weight2
    d[-1] = 1

    l = np.zeros(n)
    l[1:-1] = dr[1:] * weight1 - dr[1:]**2 * weight2

    u = np.zeros(n)
    u[0] = 2/dr[0]**2
    u[1:-1] = dr[:-1] * weight1 + dr[:-1]**2 * weight2

    # weight1 = 2/(dr[:-1]**2 + dr[1:]**2)
    # weight2 = 1/(r[1:-1] * (r[2:] - r[:-2]))

    # d = np.zeros(n)
    # d[0] = -2/dr[0]**2
    # d[1:-1] = -2*weight1
    # d[-1] = 1

    # l = np.zeros(n)
    # l[1:-1] = weight1 - weight2

    # u = np.zeros(n)
    # u[0] = 2/dr[0]**2
    # u[1:-1] = weight1 + weight2

    return l, d, u


@njit(cache=True)
def radial_potential_nonuniform_grid(r, rho):
    """
    Solves the radial Poisson equation on a nonuniform grid.
    Boundary conditions are d phi/dr = 0 at r = 0 and phi(rmax) = 0

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, must be evenly spaced, with r[0] = 0, r[-1] = r_max.
    rho : np.ndarray
        <C/m^3>
        Charge density at r.

    Returns
    -------
    np.ndarray
        Potential at r.
    """
    l, d, u = _td_nonuniform_grid(r)
    rho_ = rho[:]
    rho_[-1] = 0 #Boundary condition
    phi = tdma(l, d, u, -rho/EPS_0)
    return phi


@njit(cache=True)
def heat_capacity(r, phi, q, kT):
    """
    Computes the heat capacity of an ion cloud with a given charge state and temperature inside
    the external potenital phi(r). According to Lu, Currell cloud expansion.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, must be evenly spaced, with r[0] = 0, r[-1] = r_max.
    phi : np.ndarray
        <V>
        Potential at r.
    q : int
        Ion charge state.
    kT : float
        <eV>
        Ion temperature

    Returns
    -------
    float
        <eV/eV>
        Constant volume heat capacity
    """
    pot = q*(phi - phi[0])
    a = np.trapz(pot**2 * np.exp(-pot/kT) * r, r)
    b = np.trapz(pot * np.exp(-pot/kT) * r, r)
    c = np.trapz(np.exp(-pot/kT) * r, r)
    return 3/2 + 1/kT**2 * (a/c - b**2/c**2)

def boltzmann_radial_potential(r, rho_0, n, kT, q):
#https://www.particleincell.com/2012/nonlinear-poisson-solver/
    # Solves the nonlinear radial poisson equation for a dynamic charge distribution following
    # the Boltzmann law
    # A * phi = b0 + bx (where b0 and bx are the static and dynamic terms)
    # Define cost function f = A * phi - b0 - bx
    # Compute jacobian J = A - diag(d bx_i / d phi_i)
    # Solve J y = f
    # Next guess: phi = phi - y
    # Iterate until adjustment is small
    rho_0 = rho_0[:]
    rho_0[-1] = 0 #Boundary condition
    b0 = - rho_0/EPS_0 # static rhs term


    l, d, u = _td_nonuniform_grid(r) # Set up tridiagonal system
    A = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)
    phi = radial_potential_nonuniform_grid(r, rho_0) # compute static potential


    # if n.size > 1:
    #     n = np.atleast_2d(n).T
    #     kT = np.atleast_2d(kT).T
    #     q = np.atleast_2d(q).T
    #     for _ in range(20):
    #         pot = q * (phi - phi[0]) # Potential energy w.r.t. axis
    #         bx = -np.sum(n * np.exp(-pot/kT), axis=-1) * Q_E / EPS_0 # dynamic rhs term
    #         bx[-1] = 0  # boundary condition
    #         f = A.dot(phi) - (b0 + bx) # "cost function"
    #         p = np.sum(- n / kT * np.exp(-pot/kT), axis=-1) * Q_E / EPS_0
    #         y = tdma(l, d - p, u, f)
    #         phi = phi - y

    #         res = np.linalg.norm(y[:-1]/phi[:-1])
    #         if res < 1e-6:
    #             break
    if n.size == 1:
        for _ in range(20):
            pot = q * (phi - phi[0]) # Potential energy w.r.t. axis
            bx = -n * np.exp(-pot/kT) * Q_E / EPS_0 # dynamic rhs term
            bx[-1] = 0  # boundary condition
            f = A.dot(phi) - (b0 + bx) # "cost function"
            p =  n / kT * np.exp(-pot/kT) * Q_E / EPS_0
            y = tdma(l, d - p, u, f)
            phi = phi - y

            res = np.linalg.norm(y[:-1]/phi[:-1])
            if res < 1e-6:
                break

    return phi
