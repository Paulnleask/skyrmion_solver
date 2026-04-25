"""
Host side observable wrappers for energy and baryon number calculations.

Examples
--------
Use ``compute_energy`` to evaluate the total energy from the current field configuration.
Use ``compute_skyrmion_number`` to evaluate the total baryon number from the current field configuration.
Use ``compute_center_of_mass`` to evaluate the baryon-density centre of mass.
Use ``compute_rms_radius`` to evaluate the Skyrmion RMS radius.
"""

from __future__ import annotations
import math
import numpy as np
from numba import cuda
from skyrmion_solver.core.utils import compute_sum
from skyrmion_solver.core.utils import launch_3d
from skyrmion_solver.theories.berger_skyrme.kernels import compute_center_of_mass_kernel, compute_energy_kernel, compute_rms_radius_kernel, compute_skyrmion_number_kernel

def _sum_scalar_field(arr, entmp, gridsum_partial, dim_grid: int) -> float:
    """
    Reduce a flattened scalar device field to a scalar sum.

    Parameters
    ----------
    arr : device array
        Device array containing the values to sum.
    entmp : device array
        Device scratch array for the reduction.
    gridsum_partial : device array
        Partial reduction buffer.
    dim_grid : int
        Number of lattice sites.

    Returns
    -------
    float
        Sum over all sites.
    """
    entmp.copy_to_device(arr)
    out = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    arr[:] = 0.0
    return out

def _reduce_tensor9(obs9, tensor_scratch, entmp, gridsum_partial, p_i_h) -> np.ndarray:
    """
    Reduce a nine-component flattened device tensor field to a 3x3 host tensor.

    Parameters
    ----------
    obs9 : device array
        Flattened tensor field with shape ``9 * dim_grid``.
    tensor_scratch : device array
        Scalar scratch buffer of shape ``dim_grid``.
    entmp : device array
        Scalar scratch buffer for the reduction.
    gridsum_partial : device array
        Partial reduction buffer.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    ndarray
        Host tensor with shape ``(3, 3)``.
    """
    dim_grid = int(p_i_h[6])
    out = np.zeros((3, 3), dtype=np.float64)
    flat = obs9.copy_to_host().reshape(9, dim_grid)
    for a in range(9):
        tensor_scratch.copy_to_device(flat[a])
        out[a // 3, a % 3] = _sum_scalar_field(tensor_scratch, entmp, gridsum_partial, dim_grid)
    obs9[:] = 0.0
    return out

def _reduce_obs4(obs4, tensor_scratch, entmp, gridsum_partial, p_i_h) -> np.ndarray:
    """
    Reduce a four-component flattened device field to a length-4 host vector.

    Parameters
    ----------
    obs4 : device array
        Flattened field with shape ``4 * dim_grid``.
    tensor_scratch : device array
        Scalar scratch buffer of shape ``dim_grid``.
    entmp : device array
        Scalar scratch buffer for the reduction.
    gridsum_partial : device array
        Partial reduction buffer.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    ndarray
        Host vector of length 4.
    """
    dim_grid = int(p_i_h[6])
    out = np.zeros(4, dtype=np.float64)
    flat = obs4.copy_to_host().reshape(4, dim_grid)
    for a in range(4):
        tensor_scratch.copy_to_device(flat[a])
        out[a] = _sum_scalar_field(tensor_scratch, entmp, gridsum_partial, dim_grid)
    obs4[:] = 0.0
    return out

def compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h):
    """
    Compute the total energy by summing the per-site energy contributions.

    Parameters
    ----------
    Field : device array
        Device field array containing the simulation fields.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer receiving the per-site energy contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.
    p_f_h : host array
        Float host parameter array.

    Returns
    -------
    float
        Total energy over the grid.
    """
    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))
    compute_energy_kernel[grid3d, block3d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    return _sum_scalar_field(en, entmp, gridsum_partial, int(p_i_h[6]))

def compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total baryon number by summing the per-site baryon density contributions.

    Parameters
    ----------
    Field : device array
        Device field array containing the Skyrme field components.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer receiving the per-site baryon density contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    float
        Total baryon number over the grid.
    """
    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))
    compute_skyrmion_number_kernel[grid3d, block3d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    return _sum_scalar_field(en, entmp, gridsum_partial, int(p_i_h[6]))

def compute_center_of_mass(Field, grid, d1fd1x, obs4, tensor_scratch, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h) -> np.ndarray:
    """
    Compute the baryon-density centre of mass.

    Parameters
    ----------
    Field : device array
        Device field array.
    grid : device array
        Device coordinate grid.
    d1fd1x : device array
        Device first derivative buffer.
    obs4 : device array
        Device buffer of shape ``4 * dim_grid`` for ``(B, xB, yB, zB)``.
    tensor_scratch : device array
        Device scratch array of shape ``dim_grid``.
    entmp : device array
        Device reduction scratch array.
    gridsum_partial : device array
        Device partial reduction buffer.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    ndarray
        Centre-of-mass vector of shape ``(3,)``.
    """
    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))
    compute_center_of_mass_kernel[grid3d, block3d](obs4, grid, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    moments = _reduce_obs4(obs4, tensor_scratch, entmp, gridsum_partial, p_i_h)

    B = moments[0]
    if abs(B) < 1.0e-30:
        return np.zeros(3, dtype=np.float64)
    return moments[1:4] / B

def compute_rms_radius(Field, grid, d1fd1x, en, entmp, gridsum_partial, obs4, tensor_scratch, com_d, p_i_d, p_f_d, p_i_h) -> float:
    """
    Compute the centre-of-mass RMS radius.

    Parameters
    ----------
    Field : device array
        Device field array.
    grid : device array
        Device coordinate grid.
    d1fd1x : device array
        Device first derivative buffer.
    en : device array
        Device scalar output buffer.
    entmp : device array
        Device reduction scratch array.
    gridsum_partial : device array
        Device partial reduction buffer.
    obs4 : device array
        Device ``(B, xB, yB, zB)`` moment buffer.
    tensor_scratch : device array
        Device scalar scratch array.
    com_d : device array
        Length-3 device array storing the centre of mass.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    float
        RMS radius.
    """
    com = compute_center_of_mass(Field, grid, d1fd1x, obs4, tensor_scratch, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)
    com_d.copy_to_device(com)

    B = compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)
    if abs(B) < 1.0e-30:
        return 0.0

    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))
    compute_rms_radius_kernel[grid3d, block3d](en, grid, com_d, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    numerator = _sum_scalar_field(en, entmp, gridsum_partial, int(p_i_h[6]))
    return math.sqrt(max(numerator / B, 0.0))