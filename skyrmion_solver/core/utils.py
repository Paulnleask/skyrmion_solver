"""
Core CUDA utilities for three dimensional indexing, launch configuration, and simple reductions.

Usage
-----
Use ``idx_field``, ``idx_d1``, and ``idx_d2`` inside CUDA kernels to map multi index access to flattened arrays on a three dimensional lattice.
Use ``launch_3d`` to construct a three dimensional CUDA launch configuration.
Use ``compute_sum``, ``compute_max``, and ``compute_min`` to reduce one dimensional device arrays to scalars.

Output
------
This module provides device helpers, CUDA kernels, launch helpers, and host side reduction wrappers for three dimensional CUDA workflows.
"""

from __future__ import annotations
from numba import cuda, float64

MAX_TPB = 1024

@cuda.jit(device=True, inline=True)
def idx_field(a, i, j, k, p_i):
    """
    Map a field component and lattice site to a flattened field index.

    Usage
    -----
    Use ``q = idx_field(a, x, y, z, p_i)`` to access a flattened field buffer inside a CUDA kernel.

    Parameters
    ----------
    a : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    k : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions.

    Output
    ------
    int
        Flattened index into a field like array.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    return k + j * zlen + i * ylen * zlen + a * xlen * ylen * zlen

@cuda.jit(device=True, inline=True)
def idx_d1(coord, field, i, j, k, p_i):
    """
    Map a coordinate, field component, and lattice site to a flattened first derivative index.

    Usage
    -----
    Use ``q = idx_d1(coord, a, x, y, z, p_i)`` to access a first derivative buffer inside a CUDA kernel.

    Parameters
    ----------
    coord : int
        Coordinate index.
    field : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    k : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions and coordinate count.

    Output
    ------
    int
        Flattened index into a first derivative buffer.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    number_coordinates = p_i[4]
    site_stride = xlen * ylen * zlen
    return k + j * zlen + i * ylen * zlen + coord * site_stride + field * site_stride * number_coordinates

@cuda.jit(device=True, inline=True)
def idx_d2(coord1, coord2, field, i, j, k, p_i):
    """
    Map two coordinates, a field component, and a lattice site to a flattened second derivative index.

    Usage
    -----
    Use ``q = idx_d2(c1, c2, a, x, y, z, p_i)`` to access a second derivative buffer inside a CUDA kernel.

    Parameters
    ----------
    coord1 : int
        First coordinate index.
    coord2 : int
        Second coordinate index.
    field : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    k : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions and coordinate count.

    Output
    ------
    int
        Flattened index into a second derivative buffer.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    number_coordinates = p_i[4]
    site_stride = xlen * ylen * zlen
    return k + j * zlen + i * ylen * zlen + coord1 * site_stride + coord2 * site_stride * number_coordinates + field * site_stride * number_coordinates * number_coordinates

@cuda.jit(device=True, inline=True)
def in_bounds(x, y, z, p_i):
    """
    Check whether a lattice site lies inside the active grid.

    Usage
    -----
    Use ``if not in_bounds(x, y, z, p_i): return`` to guard CUDA kernel work outside the lattice.

    Parameters
    ----------
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions.

    Output
    ------
    bool
        ``True`` if the site lies inside the lattice and ``False`` otherwise.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    return (0 <= x < xlen) and (0 <= y < ylen) and (0 <= z < zlen)

@cuda.jit
def set_field_zero_kernel(Field, p_i):
    """
    Set all field components to zero at each lattice site.

    Usage
    -----
    Launch ``set_field_zero_kernel[grid3d, block3d](Field, p_i)`` to zero a device field buffer.

    Parameters
    ----------
    Field : device array
        Flattened field buffer.
    p_i : device array
        Integer parameter array containing the grid dimensions and field count.

    Output
    ------
    None
        The field buffer is set to zero in place.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        Field[idx_field(a, x, y, z, p_i)] = 0.0

def launch_3d(p_i_h, threads=(8, 8, 8)):
    """
    Compute a three dimensional CUDA launch configuration covering the full lattice.

    Usage
    -----
    Use ``grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 8))`` before launching a three dimensional CUDA kernel.

    Parameters
    ----------
    p_i_h : host array
        Integer parameter array containing the lattice dimensions.
    threads : tuple, optional
        Threads per block given as ``(bx, by, bz)``.

    Output
    ------
    tuple
        Pair ``(grid3d, block3d)`` suitable for kernels using ``cuda.grid(3)``.
    """
    xlen = int(p_i_h[0]); ylen = int(p_i_h[1]); zlen = int(p_i_h[2])
    bx, by, bz = threads
    grid = ((xlen + bx - 1) // bx, (ylen + by - 1) // by, (zlen + bz - 1) // bz)
    return grid, (bx, by, bz)

@cuda.jit
def reduce_sum_kernel(var, partial, size):
    """
    Compute block level partial sums for a one dimensional device array.

    Usage
    -----
    Launch ``reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial sums on the device.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial sum per block.
    size : int
        Number of valid entries in ``var``.

    Output
    ------
    None
        Partial sums are written into ``partial``.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    v = 0.0
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def reduce_max_kernel(var, partial, size):
    """
    Compute block level partial maxima for a one dimensional device array.

    Usage
    -----
    Launch ``reduce_max_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial maxima on the device.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial maximum per block.
    size : int
        Number of valid entries in ``var``.

    Output
    ------
    None
        Partial maxima are written into ``partial``.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    v = -1.0e300
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride and sdata[tid] < sdata[tid + stride]:
            sdata[tid] = sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def reduce_min_kernel(var, partial, size):
    """
    Compute block level partial minima for a one dimensional device array.

    Usage
    -----
    Launch ``reduce_min_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial minima on the device.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial minimum per block.
    size : int
        Number of valid entries in ``var``.

    Output
    ------
    None
        Partial minima are written into ``partial``.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    v = 1.0e300
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride and sdata[tid] > sdata[tid + stride]:
            sdata[tid] = sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

def compute_sum(var_d, partial_d, size: int) -> float:
    """
    Compute the sum of a one dimensional device array.

    Usage
    -----
    Use ``total = compute_sum(var_d, partial_d, size)`` to reduce a device array to a scalar sum.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial sums.
    size : int
        Number of valid entries in ``var_d``.

    Output
    ------
    float
        Sum of the array entries.
    """
    tpb = 1024
    blocks = (size + tpb - 1) // tpb
    reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host()[:blocks].sum())

def compute_max(var_d, partial_d, size: int) -> float:
    """
    Compute the maximum of a one dimensional device array.

    Usage
    -----
    Use ``m = compute_max(var_d, partial_d, size)`` to reduce a device array to its maximum value.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial maxima.
    size : int
        Number of valid entries in ``var_d``.

    Output
    ------
    float
        Maximum array entry.
    """
    tpb = 1024
    blocks = (size + tpb - 1) // tpb
    reduce_max_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host()[:blocks].max())

def compute_min(var_d, partial_d, size: int) -> float:
    """
    Compute the minimum of a one dimensional device array.

    Usage
    -----
    Use ``m = compute_min(var_d, partial_d, size)`` to reduce a device array to its minimum value.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial minima.
    size : int
        Number of valid entries in ``var_d``.

    Output
    ------
    float
        Minimum array entry.
    """
    tpb = 1024
    blocks = (size + tpb - 1) // tpb
    reduce_min_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host()[:blocks].min())

def compute_max_field(EnergyGradient, max_partial, p_i_h) -> float:
    """
    Compute the maximum entry of the flattened energy gradient field.

    Usage
    -----
    Use ``err = compute_max_field(EnergyGradient, max_partial, p_i_h)`` to compute the maximum gradient entry.

    Parameters
    ----------
    EnergyGradient : device array
        Device array containing the flattened energy gradient.
    max_partial : device array
        Device array used to store partial maxima.
    p_i_h : host array
        Integer parameter array containing the flattened field size.

    Output
    ------
    float
        Maximum entry of ``EnergyGradient``.
    """
    dim_fields = p_i_h[7]
    return compute_max(EnergyGradient, max_partial, int(dim_fields))