"""
This module contains function for solving the radial space charge problem in an EBIS.
The ions are assumed to follow a Boltzmann statistic in the radial trapping potential. Their charge
in turn also shapes the potential once the number of ions becomes significant.
"""
import numpy as np
from numba import njit

from ..physconst import EPS_0, Q_E, PI, M_E


@njit(cache=True)
def tridiagonal_matrix_algorithm(l, d, u, b):
    """
    Tridiagonal Matrix Algorithm [1]_.
    Solves a system of equations M x = b for x, where M is a tridiagonal matrix.

    M = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    Parameters
    ----------
    l : np.ndarray
        Lower diagonal vector l[i] = M[i, i-1].
    d : np.ndarray
        Diagonal vector d[i] = M[i, i].
    u : np.ndarray
        Upper diagonal vector u[i] = M[i, i+1].
    b : np.ndarray
        Inhomogenety term.

    Returns
    -------
    x : np.ndarray
        Solution of the linear system.

    References
    ----------
    .. [1] "Tridiagonal matrix algorithm"
           https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = l.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = u[0]/d[0]
    dp[0] = b[0]/d[0]
    for k in range(1, n):
        cp[k] = u[k]               /(d[k]-l[k]*cp[k-1])
        dp[k] = (b[k]-l[k]*dp[k-1])/(d[k]-l[k]*cp[k-1])
    x[-1] = dp[-1]
    for k in range(n-2, -1, -1):
        x[k] = dp[k] - cp[k]*x[k+1]
    return x


@njit(cache=True)
def fd_system_uniform_grid(r):
    """
    Sets up the three diagonal vectors for a finite Poisson problem with radial symmetry on a
    uniform grid.
    d phi/dr = 0 at r = 0, and phi = phi0 at r = (n-1) * dr = r_max

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, must be evenly spaced, with r[0] = 0, r[-1] = r_max.

    Returns
    -------
    l : np.ndarray
        Lower diagonal vector.
    d : np.ndarray
        Diagonal vector.
    u : np.ndarray
        Upper diagonal vector.

    See Also
    --------
    ebisim.simulation._radial_dist.fd_system_nonuniform_grid
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
    l, d, u = fd_system_uniform_grid(r)
    rho_ = rho[:]
    rho_[-1] = 0 #Boundary condition
    phi = tridiagonal_matrix_algorithm(l, d, u, -rho/EPS_0)
    return phi


@njit(cache=True)
def fd_system_nonuniform_grid(r):
    """
    Sets up the three diagonal vectors for a finite Poisson problem with radial symmetry on a
    nonuniform grid.
    d phi/dr = 0 at r = 0, and phi = phi0 at r = (n-1) * dr = r_max
    The finite differences are developed according to [1]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.

    Returns
    -------
    l : np.ndarray
        Lower diagonal vector.
    d : np.ndarray
        Diagonal vector.
    u : np.ndarray
        Upper diagonal vector.

    References
    ----------
    .. [1] "A simple finite-difference grid with non-constant intervals",
            Sundqvist, H., & Veronis, G.,
            Tellus, 22(1), 26–31 (1970),
            https://doi.org/10.3402/tellusa.v22i1.10155

    See Also
    --------
    ebisim.simulation._radial_dist.fd_system_uniform_grid
    """
    dr = r[1:] - r[:-1]
    n = r.size

    weight1 = 2/(dr[1:] * dr[:-1] * (dr[1:] + dr[:-1]))
    weight2 = 1/(r[1:-1]*(dr[1:]**2 * dr[:-1] + dr[1:] * dr[:-1]**2))

    d = np.zeros(n)
    d[0] = -2/dr[0]**2
    d[1:-1] = -(dr[:-1] + dr[1:]) * weight1 + (dr[1:]**2 - dr[:-1]**2) * weight2
    d[-1] = 1

    l = np.zeros(n)
    l[1:-1] = dr[1:] * weight1 - dr[1:]**2 * weight2

    u = np.zeros(n)
    u[0] = 2/dr[0]**2
    u[1:-1] = dr[:-1] * weight1 + dr[:-1]**2 * weight2

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
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    rho : np.ndarray
        <C/m^3>
        Charge density at r.

    Returns
    -------
    np.ndarray
        Potential at r.
    """
    l, d, u = fd_system_nonuniform_grid(r)
    rho_ = rho[:]
    rho_[-1] = 0 #Boundary condition
    phi = tridiagonal_matrix_algorithm(l, d, u, -rho/EPS_0)
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
        Radial grid points, with r[0] = 0, r[-1] = r_max.
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


@njit(cache=True)
def boltzmann_radial_potential_onaxis_density(r, rho_0, n, kT, q, first_guess=None, ldu=None):
    """
    Solves the Boltzmann Poisson equation for a static background charge density rho_0 and particles
    with a fixed on axis density n, Temperature kT and charge state q.

    Below, nRS and nCS are the number of radial sampling points and charge states.

    Solution is found through Newton iterations, cf. [1]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    rho_0 : np.ndarray (nRS, )
        <C/m^3>
        Static charge density at r.
    n : np.ndarray (1, nCS)
        <1/m>
        On axis density of Boltzmann distributed particles.
    kT : np.ndarray (1, nCS)
        <eV>
        Temperature of Boltzmann distributed particles.
    q : np.ndarray (1, nCS)
        Charge state of Boltzmann distributed particles.
    ldu : (np.ndarray, np.ndarray, np.ndarray)
        The lower diagonal, diagonal, and upper diagonal vector describing the finite difference
        scheme. Can be provided if they have been pre-computed.

    Returns
    -------
    phi : np.ndarray (nRS, )
        <V>
        Potential at r.
    nax : np.ndarray (1, nCS)
        <1/m^3>
        On axis number densities.
    shape : np.ndarray (nRS, nCS)
        Radial shape factor of the particle distributions.

    References
    ----------
    .. [1] "Nonlinear Poisson Solver"
           https://www.particleincell.com/2012/nonlinear-poisson-solver/
    """
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


    if ldu is not None:
        l, d, u = ldu
    else:
        l, d, u = fd_system_nonuniform_grid(r) # Set up tridiagonal system
    # A = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    if first_guess is None:
        phi = radial_potential_nonuniform_grid(r, rho_0) # compute static potential
    else:
        phi = first_guess

    n = np.atleast_2d(np.asarray(n))
    kT = np.atleast_2d(np.asarray(kT))
    q = np.atleast_2d(np.asarray(q))

    for _ in range(500):
        shape = np.exp(-q * (phi - phi[0])/kT)

        _bx = - n * q * shape * Q_E / EPS_0 # dynamic rhs term
        _bx[:, -1] = 0  # boundary condition
        bx = np.sum(_bx, axis=0)

        # f = A.dot(phi) - (b0 + bx) # Target function
        f = d * phi - (b0 + bx) # Target function
        f[:-1] += u[:-1] * phi[1:]
        f[1:] += l[1:] * phi[:-1]

        j_d = - np.sum(_bx * q/kT, axis=0) #Diagonal of the Jacobian Jacobian df/dphi_i

        y = tridiagonal_matrix_algorithm(l, d - j_d, u, f)
        res = np.linalg.norm(y)/phi.size
        if res < 1e-3:
            break
        else:
            phi = phi - y

    return phi, n, shape


@njit(cache=True)
def boltzmann_radial_potential_line_density(r, rho_0, nl, kT, q, first_guess=None, ldu=None):
    """
    Solves the Boltzmann Poisson equation for a static background charge density rho_0 and particles
    with line number density n, Temperature kT and charge state q.

    Below, nRS and nCS are the number of radial sampling points and charge states.

    Solution is found through Newton iterations, cf. [1]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    rho_0 : np.ndarray (nRS, )
        <C/m^3>
        Static charge density at r.
    nl : np.ndarray (1, nCS)
        <1/m>
        Line number density of Boltzmann distributed particles.
    kT : np.ndarray (1, nCS)
        <eV>
        Temperature of Boltzmann distributed particles.
    q : np.ndarray (1, nCS)
        Charge state of Boltzmann distributed particles.
    ldu : (np.ndarray, np.ndarray, np.ndarray)
        The lower diagonal, diagonal, and upper diagonal vector describing the finite difference
        scheme. Can be provided if they have been pre-computed.

    Returns
    -------
    phi : np.ndarray (nRS, )
        <V>
        Potential at r.
    nax : np.ndarray (1, nCS)
        <1/m^3>
        On axis number densities.
    shape : np.ndarray (nRS, nCS)
        Radial shape factor of the particle distributions.

    References
    ----------
    .. [1] "Nonlinear Poisson Solver"
           https://www.particleincell.com/2012/nonlinear-poisson-solver/
    """
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

    if ldu is not None:
        l, d, u = ldu
    else:
        l, d, u = fd_system_nonuniform_grid(r) # Set up tridiagonal system
    # A = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    if first_guess is None:
        phi = radial_potential_nonuniform_grid(r, rho_0) # compute static potential
    else:
        phi = first_guess

    nl = np.atleast_2d(np.asarray(nl))
    kT = np.atleast_2d(np.asarray(kT))
    q = np.atleast_2d(np.asarray(q))

    for _ in range(500):

        shape = np.exp(-q * (phi - phi[0])/kT)
        i_sr = np.atleast_2d(np.trapz(r*shape, r)).T
        nax = nl / 2 / PI / i_sr

        _bx = - nax * q * shape * Q_E / EPS_0 # dynamic rhs term
        _bx[:, -1] = 0  # boundary condition
        bx = np.sum(_bx, axis=0)

        # F = A.dot(phi) - (b0 + bx)
        f = d * phi - (b0 + bx) # Target function
        f[:-1] += u[:-1] * phi[1:]
        f[1:] += l[1:] * phi[:-1]

        _c = np.zeros_like(shape)
        _c[:, :-1] = r[:-1] * (r[1:]-r[:-1]) * shape[:, :-1]
        j_d = - np.sum(_bx * q/kT *(i_sr-_c)/i_sr, axis=0) #Diagonal of the Jacobian df/dphi_i

        y = tridiagonal_matrix_algorithm(l, d - j_d, u, f)
        res = np.linalg.norm(y)/phi.size
        if res < 1e-3:
            break
        else:
            phi = phi - y

    return phi, nax, shape


@njit(cache=True)
def boltzmann_radial_potential_line_density_ebeam(
        r, current, r_e, e_kin, nl, kT, q, first_guess=None, ldu=None
    ):
    """
    Solves the Boltzmann Poisson equation for a static background charge density rho_0 and particles
    with line number density n, Temperature kT and charge state q.
    The electron beam charge density is computed from a uniform current density and
    the iteratively corrected velocity profile of the electron beam.

    Below, nRS and nCS are the number of radial sampling points and charge states.

    Solution is found through Newton iterations, cf. [1]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    current : float
        <A>
        Electron beam current (positive sign).
    r_e : float
        <m>
        Electron beam radius.
    e_kin : float
        <eV>
        Uncorrected electron beam energy.
    nl : np.ndarray (1, nCS)
        <1/m>
        Line number density of Boltzmann distributed particles.
    kT : np.ndarray (1, nCS)
        <eV>
        Temperature of Boltzmann distributed particles.
    q : np.ndarray (1, nCS)
        Charge state of Boltzmann distributed particles.
    ldu : (np.ndarray, np.ndarray, np.ndarray)
        The lower diagonal, diagonal, and upper diagonal vector describing the finite difference
        scheme. Can be provided if they have been pre-computed.

    Returns
    -------
    phi : np.ndarray (nRS, )
        <V>
        Potential at r.
    nax : np.ndarray (1, nCS)
        <1/m^3>
        On axis number densities.
    shape : np.ndarray (nRS, nCS)
        Radial shape factor of the particle distributions.

    References
    ----------
    .. [1] "Nonlinear Poisson Solver"
           https://www.particleincell.com/2012/nonlinear-poisson-solver/
    """
    # Solves the nonlinear radial poisson equation for a dynamic charge distribution following
    # the Boltzmann law
    # A * phi = b0 + bx (where b0 and bx are the static and dynamic terms)
    # Define cost function f = A * phi - b0 - bx
    # Compute jacobian J = A - diag(d bx_i / d phi_i)
    # Solve J y = f
    # Next guess: phi = phi - y
    # Iterate until adjustment is small
    cden = np.zeros(r.size)
    cden[r < r_e] = -current/PI/r_e**2


    if ldu is not None:
        l, d, u = ldu
    else:
        l, d, u = fd_system_nonuniform_grid(r) # Set up tridiagonal system
    # A = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    if first_guess is None:
        phi = radial_potential_nonuniform_grid(r, cden/np.sqrt(2 * Q_E * e_kin/M_E))
    else:
        phi = first_guess

    nl = np.atleast_2d(np.asarray(nl))
    kT = np.atleast_2d(np.asarray(kT))
    q = np.atleast_2d(np.asarray(q))

    for _ in range(500):

        shape = np.exp(-q * (phi - phi[0])/kT)
        i_sr = np.atleast_2d(np.trapz(r*shape, r)).T
        nax = nl / 2 / PI / i_sr

        # dynamic rhs term
        _bx_a = - nax * q * shape * Q_E / EPS_0 # dynamic rhs term
        _bx_b = - cden/np.sqrt(2 * Q_E * (e_kin+phi)/M_E) / EPS_0
        _bx_a[:, -1] = 0  # boundary condition
        bx = np.sum(_bx_a, axis=0) + _bx_b

        # F = A.dot(phi) - (b0 + bx)
        f = d * phi - bx # Target function
        f[:-1] += u[:-1] * phi[1:]
        f[1:] += l[1:] * phi[:-1]

        _c = np.zeros_like(shape)
        _c[:, :-1] = r[:-1] * (r[1:]-r[:-1]) * shape[:, :-1]
        #Diagonal of the Jacobian df/dphi_i
        j_d = -(np.sum(_bx_a * q/kT *(i_sr-_c)/i_sr, axis=0)
                + Q_E/M_E*_bx_b/(2 * Q_E * (e_kin+phi)/M_E))#Diagonal of the Jacobian df/dphi_i

        y = tridiagonal_matrix_algorithm(l, d - j_d, u, f)
        res = np.linalg.norm(y)/phi.size
        if res < 1e-3:
            break
        else:
            phi = phi - y

    return phi, nax, shape
