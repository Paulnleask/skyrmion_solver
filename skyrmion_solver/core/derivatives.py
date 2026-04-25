"""
CUDA device helpers for 3D finite difference derivatives on flattened multi component fields.

Usage
-----
Use ``compute_derivative_first`` inside a CUDA kernel to compute first spatial derivatives of a field component on a three dimensional lattice.
Use ``compute_derivative_second`` inside a CUDA kernel to compute second spatial derivatives of a field component on a three dimensional lattice.

Output
------
This module provides CUDA device functions that write first and second derivative values into flattened derivative buffers in place.
"""

from numba import cuda
from skyrmion_solver.core.utils import idx_field, idx_d1, idx_d2

@cuda.jit(device=True)
def compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f):
    """
    Compute fourth order central first derivatives of a field component at a lattice site.

    Usage
    -----
    Call ``compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)`` from within a CUDA kernel over lattice sites.

    Parameters
    ----------
    d1fd1x : device array
        Flattened buffer storing first derivatives.
    Field : device array
        Flattened buffer storing field values.
    a : int
        Field component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing grid dimensions and halo size.
    p_f : device array
        Float parameter array containing lattice spacings.

    Output
    ------
    None
        The derivative values are written into ``d1fd1x`` in place.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    lsx = p_f[3]; lsy = p_f[4]; lsz = p_f[5]
    halo = p_i[3]

    if x > halo - 1 and x < xlen - halo:
        d1fd1x[idx_d1(0, a, x, y, z, p_i)] = ((1.0 / 12.0) * Field[idx_field(a, x - 2, y, z, p_i)] - (2.0 / 3.0) * Field[idx_field(a, x - 1, y, z, p_i)] + (2.0 / 3.0) * Field[idx_field(a, x + 1, y, z, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x + 2, y, z, p_i)]) / lsx
    else:
        d1fd1x[idx_d1(0, a, x, y, z, p_i)] = 0.0

    if y > halo - 1 and y < ylen - halo:
        d1fd1x[idx_d1(1, a, x, y, z, p_i)] = ((1.0 / 12.0) * Field[idx_field(a, x, y - 2, z, p_i)] - (2.0 / 3.0) * Field[idx_field(a, x, y - 1, z, p_i)] + (2.0 / 3.0) * Field[idx_field(a, x, y + 1, z, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x, y + 2, z, p_i)]) / lsy
    else:
        d1fd1x[idx_d1(1, a, x, y, z, p_i)] = 0.0

    if z > halo - 1 and z < zlen - halo:
        d1fd1x[idx_d1(2, a, x, y, z, p_i)] = ((1.0 / 12.0) * Field[idx_field(a, x, y, z - 2, p_i)] - (2.0 / 3.0) * Field[idx_field(a, x, y, z - 1, p_i)] + (2.0 / 3.0) * Field[idx_field(a, x, y, z + 1, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x, y, z + 2, p_i)]) / lsz
    else:
        d1fd1x[idx_d1(2, a, x, y, z, p_i)] = 0.0

@cuda.jit(device=True)
def compute_derivative_second(d2fd2x, Field, a, x, y, z, p_i, p_f):
    """
    Compute fourth order central second derivatives of a field component at a lattice site.

    Usage
    -----
    Call ``compute_derivative_second(d2fd2x, Field, a, x, y, z, p_i, p_f)`` from within a CUDA kernel over lattice sites.

    Parameters
    ----------
    d2fd2x : device array
        Flattened buffer storing second derivatives.
    Field : device array
        Flattened buffer storing field values.
    a : int
        Field component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i : device array
        Integer parameter array containing grid dimensions and halo size.
    p_f : device array
        Float parameter array containing lattice spacings and pair product spacings.

    Output
    ------
    None
        The derivative values are written into ``d2fd2x`` in place.
    """
    xlen = p_i[0]; ylen = p_i[1]; zlen = p_i[2]
    lsx = p_f[3]; lsy = p_f[4]; lsz = p_f[5]
    halo = p_i[3]
    
    if x > halo - 1 and x < xlen - halo:
        d2fd2x[idx_d2(0, 0, a, x, y, z, p_i)] = (-(1.0 / 12.0) * Field[idx_field(a, x - 2, y, z, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x - 1, y, z, p_i)] - (5.0 / 2.0) * Field[idx_field(a, x, y, z, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x + 1, y, z, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x + 2, y, z, p_i)]) / (lsx * lsx)
    else:
        d2fd2x[idx_d2(0, 0, a, x, y, z, p_i)] = 0.0

    if y > halo - 1 and y < ylen - halo:
        d2fd2x[idx_d2(1, 1, a, x, y, z, p_i)] = (-(1.0 / 12.0) * Field[idx_field(a, x, y - 2, z, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x, y - 1, z, p_i)] - (5.0 / 2.0) * Field[idx_field(a, x, y, z, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x, y + 1, z, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x, y + 2, z, p_i)]) / (lsy * lsy)
    else:
        d2fd2x[idx_d2(1, 1, a, x, y, z, p_i)] = 0.0

    if z > halo - 1 and z < zlen - halo:
        d2fd2x[idx_d2(2, 2, a, x, y, z, p_i)] = (-(1.0 / 12.0) * Field[idx_field(a, x, y, z - 2, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x, y, z - 1, p_i)] - (5.0 / 2.0) * Field[idx_field(a, x, y, z, p_i)] + (4.0 / 3.0) * Field[idx_field(a, x, y, z + 1, p_i)] - (1.0 / 12.0) * Field[idx_field(a, x, y, z + 2, p_i)]) / (lsz * lsz)
    else:
        d2fd2x[idx_d2(2, 2, a, x, y, z, p_i)] = 0.0

    if (x > halo) and (y > halo) and (x < xlen - halo) and (y < ylen - halo):
        dxy = 0.0
        coeff = 1.0 / (144.0 * lsx * lsy)
        offsets = (-2, -1, 0, 1, 2)
        coeffs = (1.0, -8.0, 0.0, 8.0, -1.0)
        for i in range(5):
            ci = coeffs[i]
            xi = x + offsets[i]
            for j in range(5):
                cj = coeffs[j]
                yj = y + offsets[j]
                dxy += ci * cj * Field[idx_field(a, xi, yj, z, p_i)]
        d2fd2x[idx_d2(1, 0, a, x, y, z, p_i)] = coeff * dxy
    else:
        d2fd2x[idx_d2(1, 0, a, x, y, z, p_i)] = 0.0

    d2fd2x[idx_d2(0, 1, a, x, y, z, p_i)] = d2fd2x[idx_d2(1, 0, a, x, y, z, p_i)]

    if (x > halo) and (z > halo) and (x < xlen - halo) and (z < zlen - halo):
        dxz = 0.0
        coeff = 1.0 / (144.0 * lsx * lsz)
        offsets = (-2, -1, 0, 1, 2)
        coeffs = (1.0, -8.0, 0.0, 8.0, -1.0)
        for i in range(5):
            ci = coeffs[i]
            xi = x + offsets[i]
            for k in range(5):
                ck = coeffs[k]
                zk = z + offsets[k]
                dxz += ci * ck * Field[idx_field(a, xi, y, zk, p_i)]
        d2fd2x[idx_d2(2, 0, a, x, y, z, p_i)] = coeff * dxz
    else:
        d2fd2x[idx_d2(2, 0, a, x, y, z, p_i)] = 0.0

    d2fd2x[idx_d2(0, 2, a, x, y, z, p_i)] = d2fd2x[idx_d2(2, 0, a, x, y, z, p_i)]

    if (y > halo) and (z > halo) and (y < ylen - halo) and (z < zlen - halo):
        dyz = 0.0
        coeff = 1.0 / (144.0 * lsy * lsz)
        offsets = (-2, -1, 0, 1, 2)
        coeffs = (1.0, -8.0, 0.0, 8.0, -1.0)
        for j in range(5):
            cj = coeffs[j]
            yj = y + offsets[j]
            for k in range(5):
                ck = coeffs[k]
                zk = z + offsets[k]
                dyz += cj * ck * Field[idx_field(a, x, yj, zk, p_i)]
        d2fd2x[idx_d2(2, 1, a, x, y, z, p_i)] = coeff * dyz
    else:
        d2fd2x[idx_d2(2, 1, a, x, y, z, p_i)] = 0.0

    d2fd2x[idx_d2(1, 2, a, x, y, z, p_i)] = d2fd2x[idx_d2(2, 1, a, x, y, z, p_i)]