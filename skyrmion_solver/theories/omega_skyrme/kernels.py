"""
Purpose
-------
Core CUDA kernels and device helpers for the three dimensional non-linear sigma model coupled to the omega-meson, with the standard pion mass potential.

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
    number_skyrme_fields = p_i[11]
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
    number_skyrme_fields = p_i[11]
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
    The implementation assumes the theory specific parameters are appended to the core float array as ``mass = p_f[8]`` and ``c_omega = p_f[9]``.

    Examples
    --------
    Use ``compute_energy_point(Field, d1fd1x, x, y, z, p_i, p_f)`` inside a CUDA kernel after the first derivatives have been computed.
    """
    grid_volume = p_f[6]
    mass = p_f[8]
    c_omega = p_f[9]
    mass2 = mass * mass

    s0 = Field[idx_field(0, x, y, z, p_i)]
    omega = Field[idx_field(4, x, y, z, p_i)]

    fx0 = d1fd1x[idx_d1(0, 0, x, y, z, p_i)]
    fx1 = d1fd1x[idx_d1(0, 1, x, y, z, p_i)]
    fx2 = d1fd1x[idx_d1(0, 2, x, y, z, p_i)]
    fx3 = d1fd1x[idx_d1(0, 3, x, y, z, p_i)]
    fxomega = d1fd1x[idx_d1(0, 4, x, y, z, p_i)]

    fy0 = d1fd1x[idx_d1(1, 0, x, y, z, p_i)]
    fy1 = d1fd1x[idx_d1(1, 1, x, y, z, p_i)]
    fy2 = d1fd1x[idx_d1(1, 2, x, y, z, p_i)]
    fy3 = d1fd1x[idx_d1(1, 3, x, y, z, p_i)]
    fyomega = d1fd1x[idx_d1(1, 4, x, y, z, p_i)]

    fz0 = d1fd1x[idx_d1(2, 0, x, y, z, p_i)]
    fz1 = d1fd1x[idx_d1(2, 1, x, y, z, p_i)]
    fz2 = d1fd1x[idx_d1(2, 2, x, y, z, p_i)]
    fz3 = d1fd1x[idx_d1(2, 3, x, y, z, p_i)]
    fzomega = d1fd1x[idx_d1(2, 4, x, y, z, p_i)]

    nx = _dot4(fx0, fx1, fx2, fx3, fx0, fx1, fx2, fx3)
    ny = _dot4(fy0, fy1, fy2, fy3, fy0, fy1, fy2, fy3)
    nz = _dot4(fz0, fz1, fz2, fz3, fz0, fz1, fz2, fz3)

    energy = 0.125 * (nx + ny + nz)
    energy += mass2 * 0.25 * (1.0 - s0)
    # Omega contribution
    energy += 0.5 * omega * omega
    energy += 0.5 * (fxomega * fxomega + fyomega * fyomega + fzomega * fzomega)
    # energy -= c_omega * omega * _chern_simons_current_point(Field, d1fd1x, x, y, z, p_i)
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
    return grid_volume * _chern_simons_current_point(Field, d1fd1x, x, y, z, p_i)

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
    number_skyrme_fields = p_i[11]
    for a in range(number_skyrme_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)
    en[idx_field(0, x, y, z, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, z, p_i, p_f)

@cuda.jit
def compute_potential_kernel(en, Field, p_i, p_f):
    """
    Compute the per site omega-meson density contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local baryon density contributions.
    Field : device array
        Flattened field array.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local omega-meson density contributions are written into ``en`` in place.

    Examples
    --------
    Launch ``compute_omega_field_kernel[grid3d, block3d](en, Field, p_i, p_f)`` to evaluate the omega-meson density on the grid.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_skyrme_fields = p_i[11]
    en[idx_field(0, x, y, z, p_i)] = Field[idx_field(number_skyrme_fields, x, y, z, p_i)]

@cuda.jit(device=True)
def _chern_simons_current_point(Field, d1fd1x, x, y, z, p_i):
    """
    Compute the local Chern-Simons current from the Skyrme field.

    Parameters
    ----------
    Field : device array
        Flattened field array containing ``(sigma, pi1, pi2, pi3, omega)``.
    d1fd1x : device array
        First derivative buffer for all fields.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing model dimensions.

    Returns
    -------
    float
        Local Chern-Simons current.

    Notes
    -----
    This matches the C++ implementation
    ``ChernSimonsCurrent -= epsilon4[a][b][c][d] * phi[a] * d_x phi[b] * d_y phi[c] * d_z phi[d]``
    divided by ``2 * pi^2``.
    """
    phi0 = Field[idx_field(0, x, y, z, p_i)]
    phi1 = Field[idx_field(1, x, y, z, p_i)]
    phi2 = Field[idx_field(2, x, y, z, p_i)]
    phi3 = Field[idx_field(3, x, y, z, p_i)]

    dx0 = d1fd1x[idx_d1(0, 0, x, y, z, p_i)]
    dx1 = d1fd1x[idx_d1(0, 1, x, y, z, p_i)]
    dx2 = d1fd1x[idx_d1(0, 2, x, y, z, p_i)]
    dx3 = d1fd1x[idx_d1(0, 3, x, y, z, p_i)]

    dy0 = d1fd1x[idx_d1(1, 0, x, y, z, p_i)]
    dy1 = d1fd1x[idx_d1(1, 1, x, y, z, p_i)]
    dy2 = d1fd1x[idx_d1(1, 2, x, y, z, p_i)]
    dy3 = d1fd1x[idx_d1(1, 3, x, y, z, p_i)]

    dz0 = d1fd1x[idx_d1(2, 0, x, y, z, p_i)]
    dz1 = d1fd1x[idx_d1(2, 1, x, y, z, p_i)]
    dz2 = d1fd1x[idx_d1(2, 2, x, y, z, p_i)]
    dz3 = d1fd1x[idx_d1(2, 3, x, y, z, p_i)]

    det = _det4_cols(
        phi0, phi1, phi2, phi3,
        dx0, dx1, dx2, dx3,
        dy0, dy1, dy2, dy3,
        dz0, dz1, dz2, dz3,
    )
    return (-det / (2.0 * math.pi * math.pi))

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
    Compute the local full energy gradient and apply the local velocity update.

    Parameters
    ----------
    Velocity : device array
        Velocity array updated in place.
    Field : device array
        Flattened field array containing ``(sigma, pi1, pi2, pi3, omega)``.
    EnergyGradient : device array
        Output array receiving the local energy gradient for all five components.
    d1fd1x : device array
        First derivative buffer for all fields.
    d2fd2x : device array
        Second derivative buffer for all fields.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing model dimensions.
    p_f : device array
        Float parameter array containing the time step and theory parameters.

    Returns
    -------
    None
        The local energy gradient and velocity are written in place.

    Notes
    -----
    This matches the provided C++ structure with:
    - Skyrme gradients on components ``0..3``
    - omega gradient on component ``4``
    - omega stored as ``Field[idx_field(4, x, y, z, p_i)]``
    """
    number_coordinates = p_i[4]
    number_total_fields = p_i[5]
    number_skyrme_fields = p_i[11]
    time_step = p_f[7]
    mass = p_f[8]
    c_omega = p_f[9]

    mass2 = mass * mass
    four_pi2 = (4.0 * math.pi * math.pi)

    d_skyrme = cuda.local.array((3, 4), dtype=float64)
    dd_skyrme = cuda.local.array((3, 3, 4), dtype=float64)
    g = cuda.local.array(4, dtype=float64)

    phi0 = Field[idx_field(0, x, y, z, p_i)]
    phi1 = Field[idx_field(1, x, y, z, p_i)]
    phi2 = Field[idx_field(2, x, y, z, p_i)]
    phi3 = Field[idx_field(3, x, y, z, p_i)]

    omega = Field[idx_field(4, x, y, z, p_i)]

    for mu in range(number_skyrme_fields):
        d_skyrme[0, mu] = d1fd1x[idx_d1(0, mu, x, y, z, p_i)]
        d_skyrme[1, mu] = d1fd1x[idx_d1(1, mu, x, y, z, p_i)]
        d_skyrme[2, mu] = d1fd1x[idx_d1(2, mu, x, y, z, p_i)]

        dd_skyrme[0, 0, mu] = d2fd2x[idx_d2(0, 0, mu, x, y, z, p_i)]
        dd_skyrme[0, 1, mu] = d2fd2x[idx_d2(0, 1, mu, x, y, z, p_i)]
        dd_skyrme[0, 2, mu] = d2fd2x[idx_d2(0, 2, mu, x, y, z, p_i)]

        dd_skyrme[1, 0, mu] = d2fd2x[idx_d2(1, 0, mu, x, y, z, p_i)]
        dd_skyrme[1, 1, mu] = d2fd2x[idx_d2(1, 1, mu, x, y, z, p_i)]
        dd_skyrme[1, 2, mu] = d2fd2x[idx_d2(1, 2, mu, x, y, z, p_i)]

        dd_skyrme[2, 0, mu] = d2fd2x[idx_d2(2, 0, mu, x, y, z, p_i)]
        dd_skyrme[2, 1, mu] = d2fd2x[idx_d2(2, 1, mu, x, y, z, p_i)]
        dd_skyrme[2, 2, mu] = d2fd2x[idx_d2(2, 2, mu, x, y, z, p_i)]

        g[mu] = 0.0

    d_omega0 = d1fd1x[idx_d1(0, 4, x, y, z, p_i)]
    d_omega1 = d1fd1x[idx_d1(1, 4, x, y, z, p_i)]
    d_omega2 = d1fd1x[idx_d1(2, 4, x, y, z, p_i)]

    for mu in range(number_skyrme_fields):
        if mu == 0:
            e0 = 1.0
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
        elif mu == 1:
            e0 = 0.0
            e1 = 1.0
            e2 = 0.0
            e3 = 0.0
        elif mu == 2:
            e0 = 0.0
            e1 = 0.0
            e2 = 1.0
            e3 = 0.0
        else:
            e0 = 0.0
            e1 = 0.0
            e2 = 0.0
            e3 = 1.0

        for i in range(number_coordinates):
            g[mu] -= 0.25 * dd_skyrme[i, i, mu]

            d_omega_i = d_omega0 if i == 0 else (d_omega1 if i == 1 else d_omega2)

            for j in range(number_coordinates):
                for k in range(number_coordinates):
                    if i == j or j == k or i == k:
                        continue

                    if (i == 0 and j == 1 and k == 2) or (i == 1 and j == 2 and k == 0) or (i == 2 and j == 0 and k == 1):
                        eijk = 1.0
                    else:
                        eijk = -1.0

                    dj0 = d_skyrme[j, 0]
                    dj1 = d_skyrme[j, 1]
                    dj2 = d_skyrme[j, 2]
                    dj3 = d_skyrme[j, 3]

                    dk0 = d_skyrme[k, 0]
                    dk1 = d_skyrme[k, 1]
                    dk2 = d_skyrme[k, 2]
                    dk3 = d_skyrme[k, 3]

                    cof = _det4_cols(
                        e0, e1, e2, e3,
                        phi0, phi1, phi2, phi3,
                        dj0, dj1, dj2, dj3,
                        dk0, dk1, dk2, dk3,
                    )

                    g[mu] += c_omega / four_pi2 * eijk * d_omega_i * cof

        if mu == 0:
            g[mu] -= 0.25 * mass2

    for mu in range(number_skyrme_fields):
        EnergyGradient[idx_field(mu, x, y, z, p_i)] = g[mu]

    project_orthogonal_skyrme_field(EnergyGradient, Field, x, y, z, p_i, p_f)

    dd_omega00 = d2fd2x[idx_d2(0, 0, 4, x, y, z, p_i)]
    dd_omega11 = d2fd2x[idx_d2(1, 1, 4, x, y, z, p_i)]
    dd_omega22 = d2fd2x[idx_d2(2, 2, 4, x, y, z, p_i)]

    g_omega = omega - (dd_omega00 + dd_omega11 + dd_omega22) + c_omega * _chern_simons_current_point(Field, d1fd1x, x, y, z, p_i)

    EnergyGradient[idx_field(4, x, y, z, p_i)] = g_omega

    for mu in range(number_total_fields):
        Velocity[idx_field(mu, x, y, z, p_i)] = -time_step * EnergyGradient[idx_field(mu, x, y, z, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)