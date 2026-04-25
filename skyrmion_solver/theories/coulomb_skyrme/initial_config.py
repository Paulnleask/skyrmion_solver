"""
Initial condition and ansatz CUDA kernels for coulomb_skyrme simulations.

Purpose
-------
Provide CUDA kernels and host helpers for initializing Skyrme field configurations using the packed omega-skyrme ABI defined in `params.py`.

Usage
-----
Use `initialize(...)` as the public entrypoint for theory initialization.
Use `create_ground_state_kernel(...)` to build the vacuum.
Use `create_initial_configuration_kernel(...)` to build a hedgehog or rational-map ansatz.
Use `create_rational_map_skyrmion_kernel(...)` together with `product_ansatz_kernel(...)` to assemble product ansatz configurations such as the smorgasbord mode.

Output
------
The module updates the supplied device arrays in place, including the field, velocity, and RK work buffers.
"""

from __future__ import annotations
import math
import numpy as np
from numba import cuda
from skyrmion_solver.core.utils import idx_field, in_bounds
from skyrmion_solver.theories.coulomb_skyrme.kernels import compute_norm_skyrme_field

@cuda.jit(device=True, inline=True)
def rotate_by_y(x0: float, x1: float, x2: float, alpha: float):
    """
    Rotate a three-vector around the y-axis.

    Parameters
    ----------
    x0 : float
        x component.
    x1 : float
        y component.
    x2 : float
        z component.
    alpha : float
        Rotation angle in radians.

    Returns
    -------
    tuple[float, float, float]
        Rotated vector components.

    Examples
    --------
    Use ``xr, yr, zr = rotate_by_y(x, y, z, alpha)`` inside a CUDA device function.
    """
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return ca * x0 - sa * x2, x1, sa * x0 + ca * x2


@cuda.jit(device=True, inline=True)
def rotate_by_z(x0: float, x1: float, x2: float, alpha: float):
    """
    Rotate a three-vector around the z-axis.

    Parameters
    ----------
    x0 : float
        x component.
    x1 : float
        y component.
    x2 : float
        z component.
    alpha : float
        Rotation angle in radians.

    Returns
    -------
    tuple[float, float, float]
        Rotated vector components.

    Examples
    --------
    Use ``xr, yr, zr = rotate_by_z(x, y, z, alpha)`` inside a CUDA device function.
    """
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return ca * x0 + sa * x1, -sa * x0 + ca * x1, x2


@cuda.jit(device=True, inline=True)
def rotate_xyz(x0: float, x1: float, x2: float, a0: float, a1: float, a2: float):
    """
    Apply the Euler-angle rotation convention used by the Skyrmion rational-map code.

    Parameters
    ----------
    x0 : float
        x component.
    x1 : float
        y component.
    x2 : float
        z component.
    a0 : float
        First rotation angle in radians.
    a1 : float
        Second rotation angle in radians.
    a2 : float
        Third rotation angle in radians.

    Returns
    -------
    tuple[float, float, float]
        Rotated vector components.

    Examples
    --------
    Use ``xr, yr, zr = rotate_xyz(x, y, z, a0, a1, a2)`` inside a CUDA device function.
    """
    y0, y1, y2 = rotate_by_z(x0, x1, x2, a0)
    z0, z1, z2 = rotate_by_y(y0, y1, y2, a1)
    return rotate_by_z(z0, z1, z2, a2)


@cuda.jit(device=True, inline=True)
def cadd(a0: float, a1: float, b0: float, b1: float):
    """
    Add two complex numbers.

    Parameters
    ----------
    a0 : float
        Real part of the first number.
    a1 : float
        Imaginary part of the first number.
    b0 : float
        Real part of the second number.
    b1 : float
        Imaginary part of the second number.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the sum.

    Examples
    --------
    Use ``r0, r1 = cadd(a0, a1, b0, b1)`` inside CUDA device code.
    """
    return a0 + b0, a1 + b1


@cuda.jit(device=True, inline=True)
def csub(a0: float, a1: float, b0: float, b1: float):
    """
    Subtract two complex numbers.

    Parameters
    ----------
    a0 : float
        Real part of the first number.
    a1 : float
        Imaginary part of the first number.
    b0 : float
        Real part of the second number.
    b1 : float
        Imaginary part of the second number.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the difference.

    Examples
    --------
    Use ``r0, r1 = csub(a0, a1, b0, b1)`` inside CUDA device code.
    """
    return a0 - b0, a1 - b1


@cuda.jit(device=True, inline=True)
def cmul(a0: float, a1: float, b0: float, b1: float):
    """
    Multiply two complex numbers.

    Parameters
    ----------
    a0 : float
        Real part of the first number.
    a1 : float
        Imaginary part of the first number.
    b0 : float
        Real part of the second number.
    b1 : float
        Imaginary part of the second number.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the product.

    Examples
    --------
    Use ``r0, r1 = cmul(a0, a1, b0, b1)`` inside CUDA device code.
    """
    return a0 * b0 - a1 * b1, a0 * b1 + a1 * b0


@cuda.jit(device=True, inline=True)
def cscale(s: float, a0: float, a1: float):
    """
    Scale a complex number by a real scalar.

    Parameters
    ----------
    s : float
        Real scale factor.
    a0 : float
        Real part.
    a1 : float
        Imaginary part.

    Returns
    -------
    tuple[float, float]
        Scaled real and imaginary parts.

    Examples
    --------
    Use ``r0, r1 = cscale(s, a0, a1)`` inside CUDA device code.
    """
    return s * a0, s * a1


@cuda.jit(device=True, inline=True)
def cdiv(a0: float, a1: float, b0: float, b1: float):
    """
    Divide two complex numbers.

    Parameters
    ----------
    a0 : float
        Real part of the numerator.
    a1 : float
        Imaginary part of the numerator.
    b0 : float
        Real part of the denominator.
    b1 : float
        Imaginary part of the denominator.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the quotient.

    Examples
    --------
    Use ``r0, r1 = cdiv(a0, a1, b0, b1)`` inside CUDA device code.
    """
    den = b0 * b0 + b1 * b1 + 1.0e-18
    return (a0 * b0 + a1 * b1) / den, (a1 * b0 - a0 * b1) / den


@cuda.jit(device=True, inline=True)
def cpow_uint(z0: float, z1: float, n: int):
    """
    Raise a complex number to a nonnegative integer power.

    Parameters
    ----------
    z0 : float
        Real part of the base.
    z1 : float
        Imaginary part of the base.
    n : int
        Nonnegative integer exponent.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the power.

    Examples
    --------
    Use ``r0, r1 = cpow_uint(z0, z1, n)`` inside CUDA device code.
    """
    r0 = 1.0
    r1 = 0.0
    for _ in range(n):
        r0, r1 = cmul(r0, r1, z0, z1)
    return r0, r1


@cuda.jit(device=True, inline=True)
def zton(z0: float, z1: float):
    """
    Map a stereographic coordinate to a unit three-vector on ``S^2``.

    Parameters
    ----------
    z0 : float
        Real part of the stereographic coordinate.
    z1 : float
        Imaginary part of the stereographic coordinate.

    Returns
    -------
    tuple[float, float, float]
        Unit three-vector components.

    Examples
    --------
    Use ``nx, ny, nz = zton(z0, z1)`` inside CUDA device code.
    """
    zsq = z0 * z0 + z1 * z1
    den = 1.0 + zsq
    return 2.0 * z0 / den, 2.0 * z1 / den, (1.0 - zsq) / den


@cuda.jit(device=True, inline=True)
def _box_reference_size(p_f):
    """
    Return the reference box size used to scale the initial Skyrmion size.

    Parameters
    ----------
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Reference box size.

    Notes
    -----
    The initial Skyrmion size is taken to be a fixed fraction of the smallest box dimension so that the configuration fits inside anisotropic boxes and scales proportionally when the overall box size changes.
    """
    xsize = p_f[0]
    ysize = p_f[1]
    zsize = p_f[2]

    box_size = xsize
    if ysize < box_size:
        box_size = ysize
    if zsize < box_size:
        box_size = zsize
    return box_size


@cuda.jit(device=True, inline=True)
def _initial_profile_scale(bfloat: float, p_f):
    """
    Return the physical profile scale used by the initial configuration.

    Parameters
    ----------
    bfloat : float
        Requested baryon number parameter.
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Physical profile scale.

    Notes
    -----
    This makes the Skyrmion size a fixed fraction of the physical box size.
    If the box size is multiplied by a factor, then the initial Skyrmion size is multiplied by the same factor.
    """
    box_size = _box_reference_size(p_f)
    scale = max(abs(bfloat), 1.0)
    skyrmion_size_fraction = 0.4
    return max(skyrmion_size_fraction * box_size / scale, 1.0e-6)


@cuda.jit(device=True, inline=True)
def scaled_profile_radius(r: float, bfloat: float, p_f):
    """
    Convert a physical radius into the dimensionless profile radius used by ``profilefun``.

    Parameters
    ----------
    r : float
        Physical radius in simulation units.
    bfloat : float
        Requested baryon number parameter.
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Dimensionless profile radius.
    """
    r0 = _initial_profile_scale(bfloat, p_f)
    return r / r0


@cuda.jit(device=True, inline=True)
def ffun(r: float, a: float, b: float, c: float, d: float, e: float):
    """
    Evaluate the fitted profile helper polynomial used by ``profilefun``.

    Parameters
    ----------
    r : float
        Radius.
    a : float
        Fit coefficient.
    b : float
        Fit coefficient.
    c : float
        Fit coefficient.
    d : float
        Fit coefficient.
    e : float
        Fit coefficient.

    Returns
    -------
    float
        Fitted profile value.

    Examples
    --------
    Use ``f = ffun(r, a, b, c, d, e)`` inside CUDA device code.
    """
    poly = r * (a + r * (b + r * (c + r * (d + r * e))))
    return math.pi * (1.0 - math.tanh(abs(poly)))


@cuda.jit(device=True, inline=True)
def profilefun(r: float, bint: int, m0: float, m1: float):
    """
    Evaluate the fitted radial profile function used for rational-map initial data.

    Parameters
    ----------
    r : float
        Radius.
    bint : int
        Integer baryon number used to select the rational-map fit.
    m0 : float
        Profile interpolation parameter.
    m1 : float
        Reserved profile parameter.

    Returns
    -------
    float
        Profile value ``f(r)``.

    Examples
    --------
    Use ``f = profilefun(r, bint, m0, m1)`` inside CUDA device code.
    """
    f = 0.0
    fm = 0.0
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    e = 0.0

    if r < 1.0e-6:
        return math.pi
    if r > 15.0:
        return 0.0

    if m0 < 1.0:
        if bint == 1:
            a, b, c, d, e = 0.630647, 0.0218873, -0.0245431, 0.00300978, -0.00010542
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = 0.839091, -0.148868, 0.222428, -0.0900171, 0.0107864
        elif bint == 2:
            a, b, c, d, e = 0.162388, 0.297889, -0.0878554, 0.00931881, -0.000317398
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = 0.219378, 0.485652, -0.154501, 0.0197873, -0.000774393
        elif bint == 3:
            a, b, c, d, e = -0.000352877, 0.324957, -0.0792377, 0.00740042, -0.000232486
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = 0.0354001, 0.502948, -0.13181, 0.0142808, -0.000509656
        elif bint == 4:
            a, b, c, d, e = -0.0469393, 0.294323, -0.0599714, 0.00457823, -0.000110399
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.0127361, 0.4427, -0.0941955, 0.00808476, -0.00024485
        elif bint == 5:
            a, b, c, d, e = -0.0575367, 0.187057, -0.00254209, -0.00580857, 0.000517839
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.0568381, 0.337281, -0.0167685, -0.010309, 0.00119672
        elif bint == 6:
            a, b, c, d, e = -0.0450119, 0.104984, 0.0347624, -0.0119037, 0.000864464
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.05592, 0.229393, 0.0498254, -0.0257167, 0.00248365
        elif bint == 7:
            a, b, c, d, e = -0.0383968, 0.0703267, 0.0464174, -0.0130402, 0.000887014
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.0575166, 0.196602, 0.056918, -0.0244016, 0.00219296
        elif bint == 8:
            a, b, c, d, e = -0.013271, -0.00611001, 0.0751757, -0.0170388, 0.0010835
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.0335398, 0.0655383, 0.135378, -0.0430527, 0.00381564
        elif bint == 9:
            a, b, c, d, e = 0.00853506, -0.0576792, 0.0896121, -0.0181863, 0.00109042
            f = ffun(r, a, b, c, d, e)
            a, b, c, d, e = -0.0101412, -0.0285699, 0.181351, -0.0517179, 0.00441035
        fm = ffun(r, a, b, c, d, e)
        _ = m1
        return (1.0 - m0) * f + m0 * fm

    if bint == 1:
        a, b, c, d, e = 0.839091, -0.148868, 0.222428, -0.0900171, 0.0107864
    elif bint == 2:
        a, b, c, d, e = 0.219378, 0.485652, -0.154501, 0.0197873, -0.000774393
    elif bint == 3:
        a, b, c, d, e = 0.0354001, 0.502948, -0.13181, 0.0142808, -0.000509656
    elif bint == 4:
        a, b, c, d, e = -0.0127361, 0.4427, -0.0941955, 0.00808476, -0.00024485
    elif bint == 5:
        a, b, c, d, e = -0.0568381, 0.337281, -0.0167685, -0.010309, 0.00119672
    elif bint == 6:
        a, b, c, d, e = -0.05592, 0.229393, 0.0498254, -0.0257167, 0.00248365
    elif bint == 7:
        a, b, c, d, e = -0.0575166, 0.196602, 0.056918, -0.0244016, 0.00219296
    elif bint == 8:
        a, b, c, d, e = -0.0335398, 0.0655383, 0.135378, -0.0430527, 0.00381564
    elif bint == 9:
        a, b, c, d, e = -0.0101412, -0.0285699, 0.181351, -0.0517179, 0.00441035

    fm = ffun(r, a, b, c, d, e)
    _ = m1
    return fm


@cuda.jit(device=True, inline=True)
def rational_map_value(z0: float, z1: float, bint: int):
    """
    Evaluate the rational map ``R(z)`` for a selected baryon number.

    Parameters
    ----------
    z0 : float
        Real part of the stereographic coordinate.
    z1 : float
        Imaginary part of the stereographic coordinate.
    bint : int
        Integer baryon number selector.

    Returns
    -------
    tuple[float, float]
        Real and imaginary parts of the rational-map value.

    Notes
    -----
    For unsupported values the function falls back to ``R(z) = z^B``.

    Examples
    --------
    Use ``r0, r1 = rational_map_value(z0, z1, bint)`` inside CUDA device code.
    """
    unit0 = 1.0
    unit1 = 0.0
    ii0 = 0.0
    ii1 = 1.0
    eps0 = 1.0e-9
    eps1 = 0.0

    if bint == 1:
        return z0, z1
    if bint == 2:
        return cmul(z0, z1, z0, z1)
    if bint == 3:
        a = 1.0
        z2_0, z2_1 = cmul(z0, z1, z0, z1)
        num0, num1 = cscale(a * math.sqrt(3.0), ii0, ii1)
        num0, num1 = cmul(num0, num1, z2_0, z2_1)
        num0, num1 = csub(num0, num1, unit0, unit1)
        den0, den1 = csub(z2_0, z2_1, a * math.sqrt(3.0) * ii0, a * math.sqrt(3.0) * ii1)
        den0, den1 = cmul(z0, z1, den0, den1)
        den0, den1 = cadd(den0, den1, eps0, eps1)
        return cdiv(num0, num1, den0, den1)
    if bint == 4:
        a = 1.0
        z4_0, z4_1 = cpow_uint(z0, z1, 4)
        t0, t1 = cscale(2.0 * math.sqrt(3.0), ii0, ii1)
        t0, t1 = cmul(t0, t1, z0, z1)
        t0, t1 = cmul(t0, t1, z0, z1)
        num0, num1 = cadd(z4_0, z4_1, t0, t1)
        num0, num1 = cadd(num0, num1, unit0, unit1)
        den0, den1 = csub(z4_0, z4_1, t0, t1)
        den0, den1 = cadd(den0, den1, unit0, unit1)
        num0, num1 = cscale(a, num0, num1)
        return cdiv(num0, num1, den0, den1)
    if bint == 5:
        a = 3.0660
        b = 3.9334
        z4_0, z4_1 = cpow_uint(z0, z1, 4)
        z2_0, z2_1 = cmul(z0, z1, z0, z1)
        num0, num1 = cadd(z4_0, z4_1, b * z2_0, b * z2_1)
        num0, num1 = cadd(num0, num1, a, 0.0)
        num0, num1 = cmul(z0, z1, num0, num1)
        den0, den1 = cscale(a, z4_0, z4_1)
        den0, den1 = csub(den0, den1, b * z2_0, b * z2_1)
        den0, den1 = cadd(den0, den1, unit0, unit1)
        return cdiv(num0, num1, den0, den1)
    if bint == 6:
        a = 0.1585
        z4_0, z4_1 = cpow_uint(z0, z1, 4)
        num0, num1 = cadd(z4_0, z4_1, a * ii0, a * ii1)
        den0, den1 = cmul(a * ii0, a * ii1, z4_0, z4_1)
        den0, den1 = cadd(den0, den1, unit0, unit1)
        z2_0, z2_1 = cmul(z0, z1, z0, z1)
        den0, den1 = cmul(z2_0, z2_1, den0, den1)
        den0, den1 = cadd(den0, den1, eps0, eps1)
        return cdiv(num0, num1, den0, den1)
    if bint == 7:
        z7_0, z7_1 = cpow_uint(z0, z1, 7)
        z5_0, z5_1 = cpow_uint(z0, z1, 5)
        z2_0, z2_1 = cmul(z0, z1, z0, z1)
        num0, num1 = csub(z7_0, z7_1, 7.0 * z5_0, 7.0 * z5_1)
        num0, num1 = csub(num0, num1, 7.0 * z2_0, 7.0 * z2_1)
        num0, num1 = csub(num0, num1, unit0, unit1)
        den0, den1 = cadd(z7_0, z7_1, 7.0 * z5_0, 7.0 * z5_1)
        den0, den1 = csub(den0, den1, 7.0 * z2_0, 7.0 * z2_1)
        den0, den1 = cadd(den0, den1, unit0, unit1)
        return cdiv(num0, num1, den0, den1)
    if bint == 8:
        a = 0.1352
        z6_0, z6_1 = cpow_uint(z0, z1, 6)
        num0, num1 = csub(z6_0, z6_1, a, 0.0)
        den0, den1 = cscale(a, z6_0, z6_1)
        den0, den1 = cadd(den0, den1, unit0, unit1)
        z2_0, z2_1 = cmul(z0, z1, z0, z1)
        den0, den1 = cmul(z2_0, z2_1, den0, den1)
        den0, den1 = cadd(den0, den1, eps0, eps1)
        return cdiv(num0, num1, den0, den1)

    return cpow_uint(z0, z1, bint)


@cuda.jit(device=True, inline=True)
def calc_rational_map_skyrmion(f: float, bint: int, z0: float, z1: float, beta0: float, beta1: float, beta2: float):
    """
    Evaluate the rational-map ansatz at one lattice site.

    Parameters
    ----------
    f : float
        Radial profile value.
    bint : int
        Integer baryon number selector.
    z0 : float
        Real part of the stereographic coordinate.
    z1 : float
        Imaginary part of the stereographic coordinate.
    beta0 : float
        First isorotation angle.
    beta1 : float
        Second isorotation angle.
    beta2 : float
        Third isorotation angle.

    Returns
    -------
    tuple[float, float, float, float]
        Components ``(sigma, pi1, pi2, pi3)`` of the rational-map field.

    Examples
    --------
    Use ``s0, s1, s2, s3 = calc_rational_map_skyrmion(...)`` inside CUDA device code.
    """
    r0, r1 = rational_map_value(z0, z1, bint)
    nx, ny, nz = zton(r0, r1)
    nx, ny, nz = rotate_xyz(nx, ny, nz, beta0, beta1, beta2)
    sf = math.sin(f)
    return math.cos(f), nx * sf, ny * sf, nz * sf


@cuda.jit(device=True, inline=True)
def quaternion_prod(a0: float, a1: float, a2: float, a3: float, b0: float, b1: float, b2: float, b3: float):
    """
    Multiply two SU(2) fields represented as unit quaternions.

    Parameters
    ----------
    a0 : float
        Scalar component of the first quaternion.
    a1 : float
        First vector component of the first quaternion.
    a2 : float
        Second vector component of the first quaternion.
    a3 : float
        Third vector component of the first quaternion.
    b0 : float
        Scalar component of the second quaternion.
    b1 : float
        First vector component of the second quaternion.
    b2 : float
        Second vector component of the second quaternion.
    b3 : float
        Third vector component of the second quaternion.

    Returns
    -------
    tuple[float, float, float, float]
        Product quaternion components.

    Examples
    --------
    Use ``c0, c1, c2, c3 = quaternion_prod(...)`` inside CUDA device code.
    """
    c0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    c1 = a0 * b1 + b0 * a1 + a2 * b3 - a3 * b2
    c2 = a0 * b2 + b0 * a2 + a3 * b1 - a1 * b3
    c3 = a0 * b3 + b0 * a3 + a1 * b2 - a2 * b1
    return c0, c1, c2, c3


@cuda.jit(device=True, inline=True)
def local_coordinates(x: int, y: int, z: int, grid, p_i):
    """
    Read the physical coordinates at a lattice site.

    Parameters
    ----------
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    grid : device array
        Flattened coordinate array.
    p_i : device array
        Integer parameter array used for indexing.

    Returns
    -------
    tuple[float, float, float]
        Physical coordinates of the lattice site.

    Examples
    --------
    Use ``gx, gy, gz = local_coordinates(x, y, z, grid, p_i)`` inside CUDA device code.
    """
    gx = grid[idx_field(0, x, y, z, p_i)]
    gy = grid[idx_field(1, x, y, z, p_i)]
    gz = grid[idx_field(2, x, y, z, p_i)]
    return gx, gy, gz


@cuda.jit(device=True, inline=True)
def centered_position(x: int, y: int, z: int, grid, p_i):
    """
    Compute centered physical coordinates relative to the middle of the box.

    Parameters
    ----------
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    grid : device array
        Flattened coordinate array.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    tuple[float, float, float]
        Centered physical coordinates.

    Examples
    --------
    Use ``rx, ry, rz = centered_position(x, y, z, grid, p_i)`` inside CUDA device code.
    """
    xlen = p_i[0]
    ylen = p_i[1]
    zlen = p_i[2]

    gx, gy, gz = local_coordinates(x, y, z, grid, p_i)
    x0 = 0.5 * (grid[idx_field(0, 0, 0, 0, p_i)] + grid[idx_field(0, xlen - 1, ylen - 1, zlen - 1, p_i)])
    y0 = 0.5 * (grid[idx_field(1, 0, 0, 0, p_i)] + grid[idx_field(1, xlen - 1, ylen - 1, zlen - 1, p_i)])
    z0 = 0.5 * (grid[idx_field(2, 0, 0, 0, p_i)] + grid[idx_field(2, xlen - 1, ylen - 1, zlen - 1, p_i)])
    return gx - x0, gy - y0, gz - z0


@cuda.jit(device=True, inline=True)
def hedgehog_field(rx: float, ry: float, rz: float, bfloat: float, rotation: float, p_i, p_f):
    """
    Evaluate a simple hedgehog-style initial field at one spatial point.

    Parameters
    ----------
    rx : float
        Centered x coordinate.
    ry : float
        Centered y coordinate.
    rz : float
        Centered z coordinate.
    bfloat : float
        Requested baryon number parameter.
    rotation : float
        Isorotation angle in radians.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    tuple[float, float, float, float]
        Components ``(sigma, pi1, pi2, pi3)`` of the hedgehog field.

    Examples
    --------
    Use ``s0, s1, s2, s3 = hedgehog_field(...)`` inside CUDA device code.
    """
    r = math.sqrt(rx * rx + ry * ry + rz * rz)
    r0 = _initial_profile_scale(bfloat, p_f)

    if r < 1.0e-12:
        return -1.0, 0.0, 0.0, 0.0

    f = math.pi * math.exp(-(r / r0) * (r / r0))
    sf = math.sin(f)
    invr = 1.0 / r

    nx = rx * invr
    ny = ry * invr
    nz = rz * invr

    px, py, pz = rotate_by_z(nx, ny, nz, rotation)
    return math.cos(f), sf * px, sf * py, sf * pz


@cuda.jit
def zero_field_kernel(Field, p_i, value: float):
    """
    Set every component of a field-like buffer to a uniform value.

    Parameters
    ----------
    Field : device array
        Flattened field-like array.
    p_i : device array
        Integer parameter array.
    value : float
        Value assigned to every entry.

    Returns
    -------
    None
        The array is updated in place.

    Examples
    --------
    Launch ``zero_field_kernel[grid3d, block3d](Field, p_i, 0.0)`` to zero a field buffer.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        Field[idx_field(a, x, y, z, p_i)] = value


@cuda.jit
def create_ground_state_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i, p_f):
    """
    Initialize the vacuum state and zero the integrator buffers.

    Parameters
    ----------
    Velocity : device array
        Velocity field.
    Field : device array
        State array containing the Skyrme and omega field components.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The field and auxiliary buffers are updated in place.

    Examples
    --------
    Launch ``create_ground_state_kernel[grid3d, block3d](...)`` to initialize the vacuum.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    Field[idx_field(0, x, y, z, p_i)] = 1.0
    Field[idx_field(1, x, y, z, p_i)] = 0.0
    Field[idx_field(2, x, y, z, p_i)] = 0.0
    Field[idx_field(3, x, y, z, p_i)] = 0.0
    Field[idx_field(4, x, y, z, p_i)] = 0.0

    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, z, p_i)] = 0.0
        k1[idx_field(a, x, y, z, p_i)] = 0.0
        k2[idx_field(a, x, y, z, p_i)] = 0.0
        k3[idx_field(a, x, y, z, p_i)] = 0.0
        k4[idx_field(a, x, y, z, p_i)] = 0.0
        l1[idx_field(a, x, y, z, p_i)] = 0.0
        l2[idx_field(a, x, y, z, p_i)] = 0.0
        l3[idx_field(a, x, y, z, p_i)] = 0.0
        l4[idx_field(a, x, y, z, p_i)] = 0.0
        Temp[idx_field(a, x, y, z, p_i)] = 0.0


@cuda.jit
def create_initial_configuration_kernel(
    Velocity,
    Field,
    grid,
    k1,
    k2,
    k3,
    k4,
    l1,
    l2,
    l3,
    l4,
    Temp,
    ansatz_hedgehog,
    ansatz_rational_map,
    baryon_number,
    isospin_rotation,
    p_i,
    p_f,
):
    """
    Initialize a Skyrme ansatz configuration and zero the integrator buffers.

    Parameters
    ----------
    Velocity : device array
        Velocity field.
    Field : device array
        State array containing the Skyrme and omega field components.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    ansatz_hedgehog : bool
        Flag selecting the hedgehog ansatz.
    ansatz_rational_map : bool
        Flag selecting the rational-map ansatz.
    baryon_number : float
        Requested baryon number parameter.
    isospin_rotation : float
        Isorotation angle in radians.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The field and auxiliary buffers are updated in place.

    Examples
    --------
    Launch ``create_initial_configuration_kernel[grid3d, block3d](...)`` to initialize the ansatz configuration.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    rx, ry, rz = centered_position(x, y, z, grid, p_i)
    bint = int(abs(baryon_number) + 0.5)
    if bint < 1:
        bint = 1

    if ansatz_rational_map:
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r < 1.0e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = math.acos(rz / r)
            phi = math.atan2(ry, rx)
        t = math.tan(0.5 * theta)
        z0 = math.cos(phi) * t
        z1 = math.sin(phi) * t
        rs = scaled_profile_radius(r, baryon_number, p_f)
        f = profilefun(rs, bint, 1.0, 0.0)
        s0, s1, s2, s3 = calc_rational_map_skyrmion(f, bint, z0, z1, 0.0, 0.0, isospin_rotation)
    elif ansatz_hedgehog:
        s0, s1, s2, s3 = hedgehog_field(rx, ry, rz, baryon_number, isospin_rotation, p_i, p_f)
    else:
        s0, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0

    Field[idx_field(0, x, y, z, p_i)] = s0
    Field[idx_field(1, x, y, z, p_i)] = s1
    Field[idx_field(2, x, y, z, p_i)] = s2
    Field[idx_field(3, x, y, z, p_i)] = s3
    Field[idx_field(4, x, y, z, p_i)] = 0.0
    compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)

    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, z, p_i)] = 0.0
        k1[idx_field(a, x, y, z, p_i)] = 0.0
        k2[idx_field(a, x, y, z, p_i)] = 0.0
        k3[idx_field(a, x, y, z, p_i)] = 0.0
        k4[idx_field(a, x, y, z, p_i)] = 0.0
        l1[idx_field(a, x, y, z, p_i)] = 0.0
        l2[idx_field(a, x, y, z, p_i)] = 0.0
        l3[idx_field(a, x, y, z, p_i)] = 0.0
        l4[idx_field(a, x, y, z, p_i)] = 0.0
        Temp[idx_field(a, x, y, z, p_i)] = 0.0


@cuda.jit
def create_rational_map_skyrmion_kernel(
    Field,
    grid,
    x0,
    y0,
    z0c,
    alpha0,
    alpha1,
    alpha2,
    beta0,
    beta1,
    beta2,
    baryon_number,
    profile_m0,
    profile_m1,
    p_i,
    p_f,
):
    """
    Write a translated and rotated rational-map Skyrmion into a field buffer.

    Parameters
    ----------
    Field : device array
        Output field buffer receiving a single rational-map Skyrmion.
    grid : device array
        Flattened coordinate array.
    x0 : float
        Physical x coordinate of the Skyrmion center.
    y0 : float
        Physical y coordinate of the Skyrmion center.
    z0c : float
        Physical z coordinate of the Skyrmion center.
    alpha0 : float
        First spatial rotation angle.
    alpha1 : float
        Second spatial rotation angle.
    alpha2 : float
        Third spatial rotation angle.
    beta0 : float
        First isorotation angle.
    beta1 : float
        Second isorotation angle.
    beta2 : float
        Third isorotation angle.
    baryon_number : float
        Requested baryon number.
    profile_m0 : float
        Profile interpolation parameter.
    profile_m1 : float
        Reserved profile parameter.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The output buffer is filled with a single rational-map configuration.

    Examples
    --------
    Launch ``create_rational_map_skyrmion_kernel[grid3d, block3d](...)`` to build a single rational-map Skyrmion.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    gx, gy, gz = local_coordinates(x, y, z, grid, p_i)
    rx = gx - x0
    ry = gy - y0
    rz = gz - z0c

    rx, ry, rz = rotate_xyz(rx, ry, rz, alpha0, alpha1, alpha2)

    r = math.sqrt(rx * rx + ry * ry + rz * rz)
    bint = int(abs(baryon_number) + 0.5)
    if bint < 1:
        bint = 1

    if r < 1.0e-6:
        theta = 0.0
        phi = 0.0
    else:
        theta = math.acos(rz / r)
        phi = math.atan2(ry, rx)

    t = math.tan(0.5 * theta)
    zz0 = math.cos(phi) * t
    zz1 = math.sin(phi) * t
    rs = scaled_profile_radius(r, baryon_number, p_f)
    f = profilefun(rs, bint, profile_m0, profile_m1)
    s0, s1, s2, s3 = calc_rational_map_skyrmion(f, bint, zz0, zz1, beta0, beta1, beta2)

    Field[idx_field(0, x, y, z, p_i)] = s0
    Field[idx_field(1, x, y, z, p_i)] = s1
    Field[idx_field(2, x, y, z, p_i)] = s2
    Field[idx_field(3, x, y, z, p_i)] = s3
    Field[idx_field(4, x, y, z, p_i)] = 0.0
    compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)


@cuda.jit
def product_ansatz_kernel(Field, Temp, p_i, p_f):
    """
    Multiply a temporary SU(2) field into an accumulated SU(2) field sitewise.

    Parameters
    ----------
    Field : device array
        In-place accumulated field.
    Temp : device array
        Temporary field multiplied into ``Field``.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The accumulated field is updated in place.

    Examples
    --------
    Launch ``product_ansatz_kernel[grid3d, block3d](Field, Temp, p_i, p_f)`` to compose a product ansatz.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    a0 = Field[idx_field(0, x, y, z, p_i)]
    a1 = Field[idx_field(1, x, y, z, p_i)]
    a2 = Field[idx_field(2, x, y, z, p_i)]
    a3 = Field[idx_field(3, x, y, z, p_i)]

    b0 = Temp[idx_field(0, x, y, z, p_i)]
    b1 = Temp[idx_field(1, x, y, z, p_i)]
    b2 = Temp[idx_field(2, x, y, z, p_i)]
    b3 = Temp[idx_field(3, x, y, z, p_i)]

    c0, c1, c2, c3 = quaternion_prod(a0, a1, a2, a3, b0, b1, b2, b3)

    Field[idx_field(0, x, y, z, p_i)] = c0
    Field[idx_field(1, x, y, z, p_i)] = c1
    Field[idx_field(2, x, y, z, p_i)] = c2
    Field[idx_field(3, x, y, z, p_i)] = c3
    Field[idx_field(4, x, y, z, p_i)] = 0.0
    compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)


@cuda.jit
def create_skyrmion_kernel(Field, grid, pxi, pxj, pxk, isospin_rotation, p_i, p_f):
    """
    Compose an additional unit hedgehog Skyrmion into the existing field.

    Parameters
    ----------
    Field : device array
        State array containing the Skyrme field components.
    grid : device array
        Flattened coordinate array.
    pxi : int
        x index of the Skyrmion center.
    pxj : int
        y index of the Skyrmion center.
    pxk : int
        z index of the Skyrmion center.
    isospin_rotation : float
        Isorotation angle in radians.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The field is updated in place.

    Examples
    --------
    Launch ``create_skyrmion_kernel[grid3d, block3d](...)`` to compose a Skyrmion into the current field.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return

    xlen = p_i[0]
    ylen = p_i[1]
    zlen = p_i[2]
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen or pxk < 0 or pxk >= zlen:
        return

    gx, gy, gz = local_coordinates(x, y, z, grid, p_i)
    cx, cy, cz = local_coordinates(pxi, pxj, pxk, grid, p_i)

    s0, s1, s2, s3 = hedgehog_field(gx - cx, gy - cy, gz - cz, 1.0, isospin_rotation, p_i, p_f)

    o0 = Field[idx_field(0, x, y, z, p_i)]
    o1 = Field[idx_field(1, x, y, z, p_i)]
    o2 = Field[idx_field(2, x, y, z, p_i)]
    o3 = Field[idx_field(3, x, y, z, p_i)]

    c0, c1, c2, c3 = quaternion_prod(o0, o1, o2, o3, s0, s1, s2, s3)

    Field[idx_field(0, x, y, z, p_i)] = c0
    Field[idx_field(1, x, y, z, p_i)] = c1
    Field[idx_field(2, x, y, z, p_i)] = c2
    Field[idx_field(3, x, y, z, p_i)] = c3
    Field[idx_field(4, x, y, z, p_i)] = 0.0
    compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)


def _zero_integrator_buffers(*buffers) -> None:
    """
    Zero a sequence of device buffers in place.

    Parameters
    ----------
    *buffers
        Device arrays with slice assignment support.

    Returns
    -------
    None
        Every supplied buffer is set to zero.

    Examples
    --------
    Use ``_zero_integrator_buffers(k1, k2, k3, k4)`` after initializing a new field configuration.
    """
    for buf in buffers:
        buf[:] = 0.0


def _flat_host(a: int, x: int, y: int, z: int, p_i_h) -> int:
    """
    Compute the flattened host index for a component and lattice site.

    Parameters
    ----------
    a : int
        Field component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    p_i_h : ndarray
        Host integer parameter array.

    Returns
    -------
    int
        Flattened host index.
    """
    xlen = int(p_i_h[0])
    ylen = int(p_i_h[1])
    zlen = int(p_i_h[2])
    return z + y * zlen + x * ylen * zlen + a * xlen * ylen * zlen


def _physical_box_center(grid_h: np.ndarray, p_i_h: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the physical center of the simulation box from the host grid.

    Parameters
    ----------
    grid_h : ndarray
        Host copy of the flattened coordinate grid.
    p_i_h : ndarray
        Host integer parameter array.

    Returns
    -------
    tuple[float, float, float]
        Physical center coordinates.
    """
    xlen = int(p_i_h[0])
    ylen = int(p_i_h[1])
    zlen = int(p_i_h[2])

    x0 = 0.5 * (float(grid_h[_flat_host(0, 0, 0, 0, p_i_h)]) + float(grid_h[_flat_host(0, xlen - 1, ylen - 1, zlen - 1, p_i_h)]))
    y0 = 0.5 * (float(grid_h[_flat_host(1, 0, 0, 0, p_i_h)]) + float(grid_h[_flat_host(1, xlen - 1, ylen - 1, zlen - 1, p_i_h)]))
    z0 = 0.5 * (float(grid_h[_flat_host(2, 0, 0, 0, p_i_h)]) + float(grid_h[_flat_host(2, xlen - 1, ylen - 1, zlen - 1, p_i_h)]))
    return x0, y0, z0


def _initialize_smorgasbord(
    *,
    Velocity,
    Field,
    grid,
    k1,
    k2,
    k3,
    k4,
    l1,
    l2,
    l3,
    l4,
    Temp,
    d1fd1x,
    p_i_d,
    p_f_d,
    p_i_h,
    p_f_h,
    grid3d,
    block3d,
    config: dict,
) -> None:
    """
    Build a Smörgåsbord/product ansatz by multiplying randomly placed rational-map unit Skyrmions.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Accumulated output field.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer used for one Skyrmion factor at a time.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : ndarray
        Integer host parameter array.
    p_f_h : ndarray
        Float host parameter array.
    grid3d : tuple
        CUDA grid configuration.
    block3d : tuple
        CUDA block configuration.
    config : dict
        Smörgåsbord configuration dictionary.

    Returns
    -------
    None
        The accumulated field and integrator buffers are initialized in place.

    Examples
    --------
    Use ``_initialize_smorgasbord(..., config=cfg)`` internally from ``initialize``.
    """
    rng = np.random.default_rng(int(config.get("seed", 0)))

    xsize = float(p_f_h[0])
    ysize = float(p_f_h[1])
    zsize = float(p_f_h[2])

    xc, yc, zc = _physical_box_center(grid.copy_to_host(), p_i_h)

    count_default = max(1, int(round(abs(float(config.get("baryon_number", p_f_h[10]))))))
    count = int(config.get("count", count_default))
    lump_baryon = int(config.get("lump_baryon", 1))
    profile_m0 = float(config.get("profile_m0", 1.0))
    profile_m1 = float(config.get("profile_m1", 0.0))

    position_scale = float(config.get("position_scale", 0.18))
    dx = position_scale * xsize
    dy = position_scale * ysize
    dz = position_scale * zsize

    positions = rng.uniform(low=[-dx, -dy, -dz], high=[dx, dy, dz], size=(count, 3))
    positions -= positions.mean(axis=0, keepdims=True)

    alpha_list = rng.uniform(0.0, 2.0 * math.pi, size=(count, 3))
    beta_list = rng.uniform(0.0, 2.0 * math.pi, size=(count, 3))

    create_ground_state_kernel[grid3d, block3d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d)

    for n in range(count):
        zero_field_kernel[grid3d, block3d](Temp, p_i_d, 0.0)

        px = float(xc + positions[n, 0])
        py = float(yc + positions[n, 1])
        pz = float(zc + positions[n, 2])

        a0 = float(alpha_list[n, 0])
        a1 = float(alpha_list[n, 1])
        a2 = float(alpha_list[n, 2])

        b0 = float(beta_list[n, 0])
        b1 = float(beta_list[n, 1])
        b2 = float(beta_list[n, 2])

        create_rational_map_skyrmion_kernel[grid3d, block3d](
            Temp,
            grid,
            px,
            py,
            pz,
            a0,
            a1,
            a2,
            b0,
            b1,
            b2,
            float(lump_baryon),
            profile_m0,
            profile_m1,
            p_i_d,
            p_f_d,
        )
        product_ansatz_kernel[grid3d, block3d](Field, Temp, p_i_d, p_f_d)

    _zero_integrator_buffers(Velocity, k1, k2, k3, k4, l1, l2, l3, l4, Temp)


def initialize(
    *,
    Velocity,
    Field,
    grid,
    k1,
    k2,
    k3,
    k4,
    l1,
    l2,
    l3,
    l4,
    Temp,
    d1fd1x=None,
    p_i_d,
    p_f_d,
    p_i_h=None,
    p_f_h=None,
    grid3d,
    block3d,
    config: dict | None = None,
):
    """
    Initialize the theory field configuration.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Field configuration updated in place.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array, optional
        Integer host parameter array.
    p_f_h : host array, optional
        Float host parameter array.
    grid3d : tuple
        CUDA grid configuration.
    block3d : tuple
        CUDA block configuration.
    config : dict or None, optional
        Configuration dictionary controlling the initialization mode and ansatz.

    Returns
    -------
    None
        The selected initialization branch is launched and updates the field in place.

    Notes
    -----
    If ``config`` does not provide overrides, the ansatz defaults are read from the packed omega-skyrme parameter arrays.
    The packed omega-skyrme ABI is assumed to satisfy
    ``p_i[11:15] = (number_skyrme_fields, hedgehog, rational_map, uniform)``
    and
    ``p_f[8:12] = (mass, c_omega, baryon_number, isospin_rotation)``.

    Supported modes include ``ground``, ``initial``, ``rational_map``, and ``smorgasbord`` / ``smorgaasbord``.

    Examples
    --------
    Use ``initialize(..., grid3d=grid3d, block3d=block3d, config=config)`` to initialize the theory state.
    """
    cfg = config or {}
    mode = str(cfg.get("mode", "initial")).lower().strip()

    if mode in ("ground", "uniform", "vacuum", "gs"):
        create_ground_state_kernel[grid3d, block3d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d)
        return

    if mode in ("smorgasbord", "smorgaasbord", "smorg"):
        if p_i_h is None or p_f_h is None:
            raise ValueError("Smorgasbord initialization requires host parameter arrays p_i_h and p_f_h.")
        _initialize_smorgasbord(
            Velocity=Velocity,
            Field=Field,
            grid=grid,
            k1=k1,
            k2=k2,
            k3=k3,
            k4=k4,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            Temp=Temp,
            d1fd1x=d1fd1x,
            p_i_d=p_i_d,
            p_f_d=p_f_d,
            p_i_h=p_i_h,
            p_f_h=p_f_h,
            grid3d=grid3d,
            block3d=block3d,
            config=cfg,
        )
        return

    ans_from_params = "hedgehog"
    if p_i_h is not None:
        if int(p_i_h[13]) != 0:
            ans_from_params = "rational_map"
        elif int(p_i_h[14]) != 0:
            ans_from_params = "uniform"
        elif int(p_i_h[12]) != 0:
            ans_from_params = "hedgehog"

    if mode in ("rational_map", "rational map", "rational-map"):
        ans = "rational_map"
    else:
        ans = str(cfg.get("ansatz", ans_from_params)).lower().strip()

    ansatz_hedgehog = ans == "hedgehog"
    ansatz_rational_map = ans in ("rational_map", "rational map", "rational-map")

    baryon_number = float(cfg.get("baryon_number", p_f_h[10] if p_f_h is not None else 1.0))
    isospin_rotation = float(cfg.get("isospin_rotation", p_f_h[11] if p_f_h is not None else 0.0))

    create_initial_configuration_kernel[grid3d, block3d](
        Velocity,
        Field,
        grid,
        k1,
        k2,
        k3,
        k4,
        l1,
        l2,
        l3,
        l4,
        Temp,
        ansatz_hedgehog,
        ansatz_rational_map,
        baryon_number,
        isospin_rotation,
        p_i_d,
        p_f_d,
    )