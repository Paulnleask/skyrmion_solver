"""
Purpose
-------
Core CUDA kernels and device helpers for the three dimensional SU(2) lightly bound Skyrme model with the standard pion mass potential.

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
from skyrmion_solver.core.integrator import make_do_gradient_step_kernel, make_do_rk4_kernel

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

@cuda.jit(device=True)
def _dot3(a0: float, a1: float, a2: float, b0: float, b1: float, b2: float) -> float:
    """
    Compute the Euclidean inner product of two three-vectors.

    Parameters
    ----------
    a0, a1, a2 : float
        Components of the first vector.
    b0, b1, b2 : float
        Components of the second vector.

    Returns
    -------
    float
        Euclidean inner product.
    """
    return a0 * b0 + a1 * b1 + a2 * b2

@cuda.jit(device=True)
def _dot4(a0, a1, a2, a3, b0, b1, b2, b3):
    """
    Compute the Euclidean inner product of two four component vectors.

    Parameters
    ----------
    a0, a1, a2, a3 : float
        Components of the first vector.
    b0, b1, b2, b3 : float
        Components of the second vector.

    Returns
    -------
    float
        Euclidean inner product of the two vectors.

    Examples
    --------
    Use ``s = _dot4(...)`` inside CUDA device code to evaluate four component contractions.
    """
    return a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3

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
    number_skyrme_fields = p_i[5]
    s = 0.0
    for a in range(number_skyrme_fields):
        v = Field[idx_field(a, x, y, z, p_i)]
        s += v * v
    s = math.sqrt(s)
    if s == 0.0:
        return
    for a in range(number_skyrme_fields):
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
    number_skyrme_fields = p_i[5]
    lm = 0.0
    for a in range(number_skyrme_fields):
        lm += func[idx_field(a, x, y, z, p_i)] * Field[idx_field(a, x, y, z, p_i)]
    for a in range(number_skyrme_fields):
        func[idx_field(a, x, y, z, p_i)] -= lm * Field[idx_field(a, x, y, z, p_i)]

do_rk4_kernel = make_do_rk4_kernel(compute_norm_skyrme_field, project_orthogonal_skyrme_field)

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, z, p_i, p_f):
    """
    Compute the local static Skyrme energy contribution at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing ``(sigma, pi1, pi2, pi3)``.
    d1fd1x : device array
        Flattened first derivative buffer for the field.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing model dimensions.
    p_f : device array
        Float parameter array containing the grid volume and model parameters.

    Returns
    -------
    float
        Cell integrated energy contribution at the lattice site.

    Notes
    -----
    The implementation assumes ``Field[0] = sigma`` and ``Field[1:4] = (pi1, pi2, pi3)``.
    The implementation assumes the theory specific parameters are appended to the core float array as ``mpi = p_f[8]`` and ``alpha = p_f[9]``.

    Examples
    --------
    Use ``compute_energy_point(Field, d1fd1x, x, y, z, p_i, p_f)`` inside a CUDA kernel after the first derivatives have been computed.
    """
    grid_volume = p_f[6]
    mpi = p_f[8]
    mpi2 = mpi * mpi
    alpha = p_f[9]

    s0 = Field[idx_field(0, x, y, z, p_i)]

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

    nx = _dot4(fx0, fx1, fx2, fx3, fx0, fx1, fx2, fx3)
    ny = _dot4(fy0, fy1, fy2, fy3, fy0, fy1, fy2, fy3)
    nz = _dot4(fz0, fz1, fz2, fz3, fz0, fz1, fz2, fz3)

    dxy = _dot4(fx0, fx1, fx2, fx3, fy0, fy1, fy2, fy3)
    dxz = _dot4(fx0, fx1, fx2, fx3, fz0, fz1, fz2, fz3)
    dyz = _dot4(fy0, fy1, fy2, fy3, fz0, fz1, fz2, fz3)

    energy = (1.0 - alpha) * (nx + ny + nz)
    energy += 0.5 * ((nx * ny - dxy * dxy) + (nx * nz - dxz * dxz) + (ny * nz - dyz * dyz))
    energy += (1.0 - alpha) * 2.0 * mpi2 * (1.0 - s0)
    energy += alpha * (1.0 - s0) * (1.0 - s0) * (1.0 - s0) * (1.0 - s0)

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
    number_skyrme_fields = p_i[5]
    for a in range(number_skyrme_fields):
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
    number_skyrme_fields = p_i[5]
    for a in range(number_skyrme_fields):
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

    number_skyrme_fields = p_i[5]
    for a in range(number_skyrme_fields):
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

    Parameters
    ----------
    Velocity : device array
        Output array receiving the local velocity update.
    Field : device array
        Flattened field array containing ``(sigma, pi1, pi2, pi3)``.
    EnergyGradient : device array
        Output array receiving the local energy gradient.
    d1fd1x : device array
        First derivative buffer.
    d2fd2x : device array
        Second derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing model dimensions.
    p_f : device array
        Float parameter array containing the time step and model parameters.

    Returns
    -------
    None
        The local energy gradient and velocity are written in place.

    Notes
    -----
    The quartic term is implemented from the standard static Skyrme energy
    ``0.5 * kappa^2 * sum_{i<j} (|d_i phi|^2 |d_j phi|^2 - (d_i phi . d_j phi)^2)``.
    The potential term is the standard pion mass term ``m_pi^2 * (1 - sigma)``.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, z, p_i, p_f)`` inside a CUDA kernel after the derivatives have been computed.
    """
    number_coordinates = p_i[4]
    time_step = p_f[7]
    mpi = p_f[8]
    mpi2 = mpi * mpi
    alpha = p_f[9]
    number_total_fields = p_i[5]

    s0 = Field[idx_field(0, x, y, z, p_i)]
    F = cuda.local.array((3, 4), dtype=float64)
    S = cuda.local.array((3, 3, 4), dtype=float64)
    g = cuda.local.array(4, dtype=float64)
    norm = cuda.local.array(3, dtype=float64)

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

    for i in range(number_coordinates):
        s = 0.0
        for a in range(number_total_fields):
            s += F[i, a] * F[i, a]
        norm[i] = s

    for a in range(number_total_fields):
        g[a] -= 2.0 * (1.0 - alpha) * (S[0, 0, a] + S[1, 1, a] + S[2, 2, a])

    for i in range(number_coordinates):
        for j in range(number_coordinates):
            if i == j:
                continue

            dnormj_di = 0.0
            ddotij_di = 0.0
            dotij = 0.0

            for b in range(number_total_fields):
                dnormj_di += 2.0 * F[j, b] * S[i, j, b]
                ddotij_di += S[i, i, b] * F[j, b] + F[i, b] * S[i, j, b]
                dotij += F[i, b] * F[j, b]

            for a in range(number_total_fields):
                g[a] -= (
                    dnormj_di * F[i, a]
                    + norm[j] * S[i, i, a]
                    - ddotij_di * F[j, a]
                    - dotij * S[i, j, a]
                )

    g[0] -= (1.0 - alpha) * 2.0 * mpi2
    g[0] -= 4.0 * alpha * (1.0 - s0) * (1.0 - s0) * (1.0 - s0)

    EnergyGradient[idx_field(0, x, y, z, p_i)] = g[0]
    EnergyGradient[idx_field(1, x, y, z, p_i)] = g[1]
    EnergyGradient[idx_field(2, x, y, z, p_i)] = g[2]
    EnergyGradient[idx_field(3, x, y, z, p_i)] = g[3]

    project_orthogonal_skyrme_field(EnergyGradient, Field, x, y, z, p_i, p_f)

    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, z, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, z, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)