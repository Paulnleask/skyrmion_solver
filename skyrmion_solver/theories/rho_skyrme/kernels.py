"""
Purpose
-------
Core CUDA kernels and device helpers for the three dimensional SU(2) Skyrme model with the standard pion mass potential.

Usage
-----
Use ``create_grid_kernel`` to construct the physical coordinate grid on the device.
Use ``compute_energy_kernel`` to evaluate the per site energy contributions.
Use ``compute_skyrmion_number_kernel`` to evaluate the per site baryon density contributions.
Use ``do_gradient_step_kernel`` to compute the local energy gradient for relaxation.

Output
------
This module provides CUDA kernels and device helpers for the static three dimensional Skyrme model in terms of a unit four component field ``phi = (sigma, pi1, pi2, pi3)``.
"""

from __future__ import annotations

import math
from numba import cuda, float64
from skyrmion_solver.core.derivatives import compute_derivative_first
from skyrmion_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds
from skyrmion_solver.core.integrator import make_do_gradient_step_kernel
from skyrmion_solver.core.integrator import make_do_rk4_kernel

@cuda.jit
def create_grid_kernel(grid, p_i, p_f):
    """
    Populate the physical coordinate grid on the device.

    Parameters
    ----------
    grid : device array
        Flattened coordinate array with components for the spatial coordinates.
    p_i : device array
        Integer parameter array used for indexing and bounds checks.
    p_f : device array
        Float parameter array containing the lattice spacings.

    Returns
    -------
    None
        The coordinate values are written into ``grid`` in place.

    Examples
    --------
    Launch ``create_grid_kernel[grid3d, block3d](grid, p_i, p_f)`` to construct the coordinate grid.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    lsx = p_f[3]; lsy = p_f[4]; lsz = p_f[5]
    grid[idx_field(0, x, y, z, p_i)] = lsx * float(x)
    grid[idx_field(1, x, y, z, p_i)] = lsy * float(y)
    grid[idx_field(2, x, y, z, p_i)] = lsz * float(z)

@cuda.jit(device=True)
def _eps3(i: int, j: int, k: int) -> float:
    """
    Return the Levi-Civita symbol in three dimensions.

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.

    Returns
    -------
    float
        ``epsilon_{ijk}``.
    """
    if i == j or j == k or i == k:
        return 0.0
    if (i == 0 and j == 1 and k == 2) or (i == 1 and j == 2 and k == 0) or (i == 2 and j == 0 and k == 1):
        return 1.0
    return -1.0

@cuda.jit(device=True, inline=True)
def idx_rho(spatial_idx: int, iso_idx: int) -> int:
    """
    Map a rho component ``rho_i^a`` to the flattened field component index.
    """
    return 4 + 3 * spatial_idx + iso_idx


@cuda.jit(device=True)
def _cross3(a0: float, a1: float, a2: float, b0: float, b1: float, b2: float, out):
    """
    Compute a three-vector cross product into ``out``.
    """
    out[0] = a1 * b2 - a2 * b1
    out[1] = a2 * b0 - a0 * b2
    out[2] = a0 * b1 - a1 * b0

@cuda.jit(device=True)
def _det3(
    a00, a01, a02,
    a10, a11, a12,
    a20, a21, a22,
):
    """
    Compute the determinant of a three by three matrix.

    Parameters
    ----------
    a00, a01, a02, a10, a11, a12, a20, a21, a22 : float
        Matrix entries in row major order.

    Returns
    -------
    float
        Determinant of the matrix.

    Examples
    --------
    Use ``d = _det3(...)`` inside CUDA device code when expanding a four by four determinant.
    """
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )

@cuda.jit(device=True)
def _det4_cols(
    c00, c10, c20, c30,
    c01, c11, c21, c31,
    c02, c12, c22, c32,
    c03, c13, c23, c33,
):
    """
    Compute the determinant of a four by four matrix given by its columns.

    Parameters
    ----------
    c00, c10, c20, c30 : float
        Components of the first column.
    c01, c11, c21, c31 : float
        Components of the second column.
    c02, c12, c22, c32 : float
        Components of the third column.
    c03, c13, c23, c33 : float
        Components of the fourth column.

    Returns
    -------
    float
        Determinant of the four by four matrix.

    Examples
    --------
    Use ``d = _det4_cols(...)`` inside CUDA device code to evaluate the baryon density Jacobian.
    """
    m00 = c00; m01 = c01; m02 = c02; m03 = c03
    m10 = c10; m11 = c11; m12 = c12; m13 = c13
    m20 = c20; m21 = c21; m22 = c22; m23 = c23
    m30 = c30; m31 = c31; m32 = c32; m33 = c33

    det0 = _det3(m11, m12, m13, m21, m22, m23, m31, m32, m33)
    det1 = _det3(m10, m12, m13, m20, m22, m23, m30, m32, m33)
    det2 = _det3(m10, m11, m13, m20, m21, m23, m30, m31, m33)
    det3 = _det3(m10, m11, m12, m20, m21, m22, m30, m31, m32)

    return m00 * det0 - m01 * det1 + m02 * det2 - m03 * det3

@cuda.jit(device=True)
def _R_a_ij(F, a: int, i: int, j: int) -> float:
    """
    Compute ``R^a_{ij} = partial_i rho^a_j - partial_j rho^a_i``.
    """
    return F[i, idx_rho(j, a)] - F[j, idx_rho(i, a)]


@cuda.jit(device=True)
def _d_R_a_ijk(S, a: int, j: int, k: int, l: int) -> float:
    """
    Compute ``partial_l R^a_{jk}``.
    """
    return S[j, l, idx_rho(k, a)] - S[k, l, idx_rho(j, a)]


@cuda.jit(device=True)
def _Omega_a_ij(F, a: int, i: int, j: int) -> float:
    """
    Compute ``Omega^a_{ij}`` in the sigma-model formulation.
    """
    temp = F[i, 0] * F[j, a + 1] - F[i, a + 1] * F[j, 0]
    for b in range(3):
        for c in range(3):
            temp += _eps3(a, b, c) * F[i, b + 1] * F[j, c + 1]
    return temp


@cuda.jit(device=True)
def _d_Omega_a_ijk(F, S, a: int, j: int, k: int, l: int) -> float:
    """
    Compute ``partial_l Omega^a_{jk}`` in the sigma-model formulation.
    """
    temp = 0.0
    temp += S[j, l, 0] * F[k, a + 1]
    temp += F[j, 0] * S[k, l, a + 1]
    temp -= S[j, l, a + 1] * F[k, 0]
    temp -= F[j, a + 1] * S[k, l, 0]
    for b in range(3):
        for c in range(3):
            temp += _eps3(a, b, c) * (S[j, l, b + 1] * F[k, c + 1] + F[j, b + 1] * S[k, l, c + 1])
    return temp

@cuda.jit(device=True)
def compute_norm_skyrme_field(Field, x, y, z, p_i, p_f):
    """
    Normalize the local Skyrme field at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing the four components ``(sigma, pi1, pi2, pi3)``.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The four field components are normalized in place.

    Examples
    --------
    Use ``compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)`` inside a CUDA kernel to enforce ``sigma^2 + pi^2 = 1``.
    """
    s = 0.0
    for a in range(4):
        v = Field[idx_field(a, x, y, z, p_i)]
        s += v * v
    s = math.sqrt(s)
    if s == 0.0:
        return
    for a in range(4):
        Field[idx_field(a, x, y, z, p_i)] /= s

@cuda.jit(device=True)
def project_orthogonal_skyrme_field(func, Field, x, y, z, p_i, p_f):
    """
    Project a local vector field orthogonally to the Skyrme field.

    Parameters
    ----------
    func : device array
        Flattened array containing the vector to project.
    Field : device array
        Flattened field array containing the local Skyrme field.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The projected vector is written into ``func`` in place.

    Examples
    --------
    Use ``project_orthogonal_skyrme_field(func, Field, x, y, z, p_i, p_f)`` inside a CUDA kernel to enforce tangency to ``S^3``.
    """
    lm = 0.0
    for a in range(4):
        lm += func[idx_field(a, x, y, z, p_i)] * Field[idx_field(a, x, y, z, p_i)]
    for a in range(4):
        func[idx_field(a, x, y, z, p_i)] -= lm * Field[idx_field(a, x, y, z, p_i)]

do_rk4_kernel = make_do_rk4_kernel(compute_norm_skyrme_field, project_orthogonal_skyrme_field)

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, z, p_i, p_f):
    """
    Compute the local static rho-Skyrme energy contribution at a lattice site.

    Notes
    -----
    This matches the uploaded paper and the uploaded C++ for ``c2 = 1`` and ``c4 = 0.25``.
    The interaction parameter read from ``p_f[11]`` must be ``c_alpha = alpha * e``.
    """
    grid_volume = p_f[6]
    mpi = p_f[8]
    mpi2 = mpi * mpi
    mrho = p_f[9]
    mrho2 = mrho * mrho
    c_alpha = p_f[10]

    sigma = Field[idx_field(0, x, y, z, p_i)]

    F = cuda.local.array((3, 13), dtype=float64)
    for a in range(p_i[5]):
        F[0, a] = d1fd1x[idx_d1(0, a, x, y, z, p_i)]
        F[1, a] = d1fd1x[idx_d1(1, a, x, y, z, p_i)]
        F[2, a] = d1fd1x[idx_d1(2, a, x, y, z, p_i)]

    energy = 0.0

    # Dirichlet term, c2 = 1
    for i in range(3):
        for mu in range(4):
            energy += F[i, mu] * F[i, mu]

    # Skyrme term, c4 = 0.25, written exactly as in the C++ loop normalization
    for i in range(3):
        for j in range(3):
            for mu in range(4):
                for nu in range(4):
                    energy += 0.5 * (
                        (F[i, mu] * F[j, nu]) * (F[i, mu] * F[j, nu])
                        - F[i, mu] * F[j, mu] * F[i, nu] * F[j, nu]
                    )

    # Standard pion mass term, c0 = 1
    energy += 2.0 * mpi2 * (1.0 - sigma)

    # Rho terms
    for a in range(3):
        for i in range(3):
            rho_ai = Field[idx_field(idx_rho(i, a), x, y, z, p_i)]
            energy += 4.0 * mrho2 * rho_ai * rho_ai
            for j in range(3):
                Rij = _R_a_ij(F, a, i, j)
                energy += 2.0 * Rij * Rij
                energy -= 8.0 * c_alpha * Rij * _Omega_a_ij(F, a, i, j)

    return energy * grid_volume

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site energy contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local energy contributions.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        First derivative buffer written during the kernel and used for the energy evaluation.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local energy contributions are written into ``en`` in place.

    Examples
    --------
    Launch ``compute_energy_kernel[grid3d, block3d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the energy density on the grid.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)
    en[idx_field(0, x, y, z, p_i)] = compute_energy_point(Field, d1fd1x, x, y, z, p_i, p_f)

@cuda.jit
def compute_rho_meson_kernel(en, Field, p_i):
    """
    Compute the per site rho-meson density across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local rho-meson contributions.
    Field : device array
        Flattened field array.
    p_i : device array
        Integer parameter array containing the number of field components.

    Returns
    -------
    None
        The local rho-meson density is written into ``en`` in place.

    Examples
    --------
    Launch ``compute_rho_meson_kernel[grid3d, block3d](en, Field, p_i)`` to evaluate the rho-meson density on the grid.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_total_fields = p_i[5]
    for a in range(4, number_total_fields):
        en[idx_field(0, x, y, z, p_i)] += Field[idx_field(a, x, y, z, p_i)] * Field[idx_field(a, x, y, z, p_i)]

@cuda.jit(device=True)
def compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f):
    """
    Compute the local baryon density contribution at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing ``(sigma, pi1, pi2, pi3)``.
    d1fd1x : device array
        First derivative buffer for the field.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the cell volume.

    Returns
    -------
    float
        Per cell contribution to the baryon number.

    Notes
    -----
    This is the degree density for the map ``phi : R^3 -> S^3`` written as the oriented four dimensional volume spanned by ``phi``, ``d_x phi``, ``d_y phi``, and ``d_z phi``.

    Examples
    --------
    Use ``compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f)`` inside a CUDA kernel after the first derivatives have been computed.
    """
    grid_volume = p_f[6]

    s0 = Field[idx_field(0, x, y, z, p_i)]
    s1 = Field[idx_field(1, x, y, z, p_i)]
    s2 = Field[idx_field(2, x, y, z, p_i)]
    s3 = Field[idx_field(3, x, y, z, p_i)]

    fx0 = d1fd1x[idx_d1(0, 0, x, y, z, p_i)]
    fx1 = d1fd1x[idx_d1(0, 1, x, y, z, p_i)]
    fx2 = d1fd1x[idx_d1(0, 2, x, y, z, p_i)]
    fx3 = d1fd1x[idx_d1(0, 3, x, y, z, p_i)]

    fy0 = d1fd1x[idx_d1(1, 0, x, y, z, p_i)]
    fy1 = d1fd1x[idx_d1(1, 1, x, y, z, p_i)]
    fy2 = d1fd1x[idx_d1(1, 2, x, y, z, p_i)]
    fy3 = d1fd1x[idx_d1(1, 3, x, y, z, p_i)]

    fz0 = d1fd1x[idx_d1(2, 0, x, y, z, p_i)]
    fz1 = d1fd1x[idx_d1(2, 1, x, y, z, p_i)]
    fz2 = d1fd1x[idx_d1(2, 2, x, y, z, p_i)]
    fz3 = d1fd1x[idx_d1(2, 3, x, y, z, p_i)]

    charge = _det4_cols(
        s0, s1, s2, s3,
        fx0, fx1, fx2, fx3,
        fy0, fy1, fy2, fy3,
        fz0, fz1, fz2, fz3,
    )
    return -charge * (grid_volume / (2.0 * math.pi * math.pi))

@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site baryon density contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local baryon density contributions.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        First derivative buffer written during the kernel and used for the baryon density evaluation.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local baryon density contributions are written into ``en`` in place.

    Examples
    --------
    Launch ``compute_skyrmion_number_kernel[grid3d, block3d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the baryon density on the grid.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    for a in range(4):
        compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)
    en[idx_field(0, x, y, z, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f)

@cuda.jit
def compute_center_of_mass_kernel(obs4, grid, Field, d1fd1x, p_i, p_f):
    """
    Compute the local baryon moments needed for the centre of mass.

    Parameters
    ----------
    obs4 : device array
        Flattened array with four components per lattice site storing ``(B, xB, yB, zB)``.
    grid : device array
        Flattened coordinate grid.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        Flattened first derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        Local moments are written into ``obs4``.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    for a in range(p_i[5]):
        compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)

    b = compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f)
    rx = grid[idx_field(0, x, y, z, p_i)]
    ry = grid[idx_field(1, x, y, z, p_i)]
    rz = grid[idx_field(2, x, y, z, p_i)]

    obs4[idx_field(0, x, y, z, p_i)] = b
    obs4[idx_field(1, x, y, z, p_i)] = rx * b
    obs4[idx_field(2, x, y, z, p_i)] = ry * b
    obs4[idx_field(3, x, y, z, p_i)] = rz * b

@cuda.jit
def compute_rms_radius_kernel(en, grid, com, Field, d1fd1x, p_i, p_f):
    """
    Compute the local RMS numerator density ``|r|^2 |B_0|``.

    Parameters
    ----------
    en : device array
        Output scalar array.
    grid : device array
        Flattened coordinate grid.
    com : device array
        Length-3 device array containing the centre of mass.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        Flattened first derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        Local RMS integrand is written into ``en``.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    for a in range(p_i[5]):
        compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)

    b = compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f)
    rx = grid[idx_field(0, x, y, z, p_i)] - com[0]
    ry = grid[idx_field(1, x, y, z, p_i)] - com[1]
    rz = grid[idx_field(2, x, y, z, p_i)] - com[2]

    en[idx_field(0, x, y, z, p_i)] = (rx * rx + ry * ry + rz * rz) * abs(b)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, z, p_i, p_f):
    """
    Compute the local energy gradient and set the local velocity update.

    Notes
    -----
    This matches the uploaded C++ implementation for:
    ``c2 = 1``, ``c4 = 0.25``, standard pion mass potential, and the rho interaction.
    The interaction parameter in ``p_f[11]`` must be ``c_alpha = alpha * e``.
    """
    time_step = p_f[7]
    mpi = p_f[8]
    mpi2 = mpi * mpi
    mrho = p_f[9]
    mrho2 = mrho * mrho
    c_alpha = p_f[10]
    number_total_fields = p_i[5]

    F = cuda.local.array((3, 13), dtype=float64)
    S = cuda.local.array((3, 3, 13), dtype=float64)
    g = cuda.local.array(13, dtype=float64)

    for a in range(number_total_fields):
        F[0, a] = d1fd1x[idx_d1(0, a, x, y, z, p_i)]
        F[1, a] = d1fd1x[idx_d1(1, a, x, y, z, p_i)]
        F[2, a] = d1fd1x[idx_d1(2, a, x, y, z, p_i)]

        S[0, 0, a] = d2fd2x[idx_d2(0, 0, a, x, y, z, p_i)]
        S[0, 1, a] = d2fd2x[idx_d2(0, 1, a, x, y, z, p_i)]
        S[0, 2, a] = d2fd2x[idx_d2(0, 2, a, x, y, z, p_i)]

        S[1, 0, a] = d2fd2x[idx_d2(1, 0, a, x, y, z, p_i)]
        S[1, 1, a] = d2fd2x[idx_d2(1, 1, a, x, y, z, p_i)]
        S[1, 2, a] = d2fd2x[idx_d2(1, 2, a, x, y, z, p_i)]

        S[2, 0, a] = d2fd2x[idx_d2(2, 0, a, x, y, z, p_i)]
        S[2, 1, a] = d2fd2x[idx_d2(2, 1, a, x, y, z, p_i)]
        S[2, 2, a] = d2fd2x[idx_d2(2, 2, a, x, y, z, p_i)]

        g[a] = 0.0

    # Skyrme field gradient: c2 = 1, c4 = 0.25
    for mu in range(4):
        for i in range(3):
            g[mu] -= 2.0 * S[i, i, mu]
            for j in range(3):
                for nu in range(4):
                    g[mu] -= 2.0 * (
                        S[i, i, mu] * (F[j, nu] * F[j, nu])
                        + (
                            F[i, mu] * (S[i, j, nu] * F[j, nu] - S[j, j, nu] * F[i, nu])
                            - S[i, j, mu] * F[i, nu] * F[j, nu]
                        )
                    )

        if mu == 0:
            g[mu] -= 2.0 * mpi2

    # Rho backreaction on the Skyrme field, exactly as in the uploaded C++
    for a in range(3):
        for i in range(3):
            for j in range(3):
                dR = _d_R_a_ijk(S, a, i, j, i)
                g[0] += 16.0 * c_alpha * F[j, a + 1] * dR
                g[a + 1] -= 16.0 * c_alpha * F[j, 0] * dR
                for b in range(3):
                    for c in range(3):
                        g[a + 1] += 16.0 * c_alpha * _eps3(b, a, c) * F[j, c + 1] * _d_R_a_ijk(S, b, i, j, i)

    # Rho field gradient, exactly as in the uploaded C++
    for a in range(3):
        for i in range(3):
            idx_ai = idx_rho(i, a)
            val = 8.0 * mrho2 * Field[idx_field(idx_ai, x, y, z, p_i)]
            for j in range(3):
                val += 8.0 * _d_R_a_ijk(S, a, i, j, j)
                val -= 16.0 * c_alpha * _d_Omega_a_ijk(F, S, a, i, j, j)
            g[idx_ai] = val

    for a in range(number_total_fields):
        EnergyGradient[idx_field(a, x, y, z, p_i)] = g[a]

    project_orthogonal_skyrme_field(EnergyGradient, Field, x, y, z, p_i, p_f)

    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, z, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, z, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)