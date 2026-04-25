"""
Time stepping and relaxation kernels for the skyrmion_solver simulation on a three dimensional lattice.

Usage
-----
Use ``make_do_gradient_step_kernel`` to build a theory specific CUDA gradient kernel.
Use ``make_do_rk4_kernel`` to build an RK4 update kernel with optional constraint projection.
Use ``do_arrested_newton_flow`` to advance the field and velocity by one arrested Newton flow step.

Output
------
This module provides CUDA kernels and host side orchestration for three dimensional gradient based time stepping and relaxation.
"""

from numba import cuda
from skyrmion_solver.core.utils import idx_field, in_bounds, launch_3d, set_field_zero_kernel, compute_max_field, compute_sum
from skyrmion_solver.core.derivatives import compute_derivative_first, compute_derivative_second

def make_do_gradient_step_kernel(do_gradient_step_point):
    """
    Create a CUDA kernel that computes spatial derivatives and applies a per site gradient update.

    Usage
    -----
    Use ``gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)`` to create the gradient kernel.

    Parameters
    ----------
    do_gradient_step_point : device function
        CUDA device function that applies the gradient update at a single lattice site.

    Output
    ------
    function
        CUDA kernel that computes derivatives and updates the energy gradient.
    """
    @cuda.jit
    def _do_gradient_step_kernel(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, p_i, p_f):
        x, y, z = cuda.grid(3)
        if not in_bounds(x, y, z, p_i):
            return
        number_total_fields = p_i[5]
        for a in range(number_total_fields):
            compute_derivative_first(d1fd1x, Field, a, x, y, z, p_i, p_f)
            compute_derivative_second(d2fd2x, Field, a, x, y, z, p_i, p_f)
        do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, z, p_i, p_f)

    return _do_gradient_step_kernel

@cuda.jit
def do_rk4_step_kernel(k_out, Velocity, l_in, Temp, Field, k_prev, factor, p_i, p_f):
    """
    Compute one RK4 position slope and the corresponding intermediate field.

    Usage
    -----
    Launch ``do_rk4_step_kernel[grid3d, block3d](k_out, Velocity, l_in, Temp, Field, k_prev, factor, p_i, p_f)`` to compute one RK4 stage.

    Parameters
    ----------
    k_out : device array
        Output buffer for the RK4 position slope.
    Velocity : device array
        Current velocity field.
    l_in : device array
        Input RK4 velocity slope.
    Temp : device array
        Output buffer for the intermediate field.
    Field : device array
        Current field configuration.
    k_prev : device array
        Previous RK4 position slope.
    factor : float
        RK4 stage factor.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the time step.

    Output
    ------
    None
        The slope and intermediate field are written in place.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    dt = p_f[7]
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        k_out[idx_field(a, x, y, z, p_i)] = dt * (Velocity[idx_field(a, x, y, z, p_i)] + factor * l_in[idx_field(a, x, y, z, p_i)])
        Temp[idx_field(a, x, y, z, p_i)] = Field[idx_field(a, x, y, z, p_i)] + factor * k_prev[idx_field(a, x, y, z, p_i)]

@cuda.jit
def do_rk4_kernel_no_constraint(Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f):
    """
    Finalize an RK4 update without constraint projection.

    Usage
    -----
    Launch ``do_rk4_kernel_no_constraint[grid3d, block3d](Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f)`` to apply the RK4 update.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Field configuration updated in place.
    k1 : device array
        First RK4 position slope.
    k2 : device array
        Second RK4 position slope.
    k3 : device array
        Third RK4 position slope.
    k4 : device array
        Fourth RK4 position slope.
    l1 : device array
        First RK4 velocity slope.
    l2 : device array
        Second RK4 velocity slope.
    l3 : device array
        Third RK4 velocity slope.
    l4 : device array
        Fourth RK4 velocity slope.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Output
    ------
    None
        The field and velocity are updated in place.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i):
        return
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, z, p_i)] += (1.0 / 6.0) * (l1[idx_field(a, x, y, z, p_i)] + 2.0 * l2[idx_field(a, x, y, z, p_i)] + 2.0 * l3[idx_field(a, x, y, z, p_i)] + l4[idx_field(a, x, y, z, p_i)])
        Field[idx_field(a, x, y, z, p_i)] += (1.0 / 6.0) * (k1[idx_field(a, x, y, z, p_i)] + 2.0 * k2[idx_field(a, x, y, z, p_i)] + 2.0 * k3[idx_field(a, x, y, z, p_i)] + k4[idx_field(a, x, y, z, p_i)])

def make_do_rk4_kernel(compute_norm_skyrme_field, project_orthogonal_skyrme_field):
    """
    Create an RK4 finalize kernel with optional unit length constraint enforcement.

    Usage
    -----
    Use ``rk4_kernel = make_do_rk4_kernel(compute_norm_skyrme_field, project_orthogonal_skyrme_field)`` to create the constrained RK4 kernel.

    Parameters
    ----------
    compute_norm_skyrme_field : device function
        CUDA device function that computes the norm at a lattice site.
    project_orthogonal_skyrme_field : device function
        CUDA device function that projects orthogonal to the field to satisfy the unit length constraint.

    Output
    ------
    function
        CUDA kernel that finalizes the RK4 update and applies the optional constraint.
    """
    @cuda.jit
    def _do_rk4_kernel(Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f):
        x, y, z = cuda.grid(3)
        if not in_bounds(x, y, z, p_i):
            return
        number_total_fields = p_i[5]
        for a in range(number_total_fields):
            Velocity[idx_field(a, x, y, z, p_i)] += (1.0 / 6.0) * (l1[idx_field(a, x, y, z, p_i)] + 2.0 * l2[idx_field(a, x, y, z, p_i)] + 2.0 * l3[idx_field(a, x, y, z, p_i)] + l4[idx_field(a, x, y, z, p_i)])
            Field[idx_field(a, x, y, z, p_i)] += (1.0 / 6.0) * (k1[idx_field(a, x, y, z, p_i)] + 2.0 * k2[idx_field(a, x, y, z, p_i)] + 2.0 * k3[idx_field(a, x, y, z, p_i)] + k4[idx_field(a, x, y, z, p_i)])
        unit_length = p_i[10]
        if unit_length:
            compute_norm_skyrme_field(Field, x, y, z, p_i, p_f)
            project_orthogonal_skyrme_field(Velocity, Field, x, y, z, p_i, p_f)

    return _do_rk4_kernel

@cuda.jit(device=True)
def arresting_criteria_point(Velocity, EnergyGradient, p_i, x, y, z):
    """
    Compute the local velocity force projection used by the arrest criterion.

    Usage
    -----
    Call ``arresting_criteria_point(Velocity, EnergyGradient, p_i, x, y, z)`` inside a CUDA kernel.

    Parameters
    ----------
    Velocity : device array
        Velocity field.
    EnergyGradient : device array
        Energy gradient field.
    p_i : device array
        Integer parameter array.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.

    Output
    ------
    float
        Dot product of velocity and energy gradient at one lattice site.
    """
    force = 0.0
    number_total_fields = p_i[5]
    for a in range(number_total_fields):
        force += Velocity[idx_field(a, x, y, z, p_i)] * EnergyGradient[idx_field(a, x, y, z, p_i)]
    return force

@cuda.jit
def arresting_criteria_kernel(Velocity, EnergyGradient, en, p_i_d):
    """
    Compute the per site arresting criterion field.

    Usage
    -----
    Launch ``arresting_criteria_kernel[grid3d, block3d](Velocity, EnergyGradient, en, p_i_d)`` before summing the criterion.

    Parameters
    ----------
    Velocity : device array
        Velocity field.
    EnergyGradient : device array
        Energy gradient field.
    en : device array
        Output buffer storing one scalar value per lattice site.
    p_i_d : device array
        Integer parameter array on the device.

    Output
    ------
    None
        The per site criterion values are written into ``en``.
    """
    x, y, z = cuda.grid(3)
    if not in_bounds(x, y, z, p_i_d):
        return
    en[idx_field(0, x, y, z, p_i_d)] = arresting_criteria_point(Velocity, EnergyGradient, p_i_d, x, y, z)

def arresting_criteria(Velocity, EnergyGradient, en, entmp, gridsum_partial, p_i_d, p_i_h):
    """
    Compute the summed arresting criterion over the full lattice.

    Usage
    -----
    Use ``force = arresting_criteria(Velocity, EnergyGradient, en, entmp, gridsum_partial, p_i_d, p_i_h)`` to evaluate the global criterion.

    Parameters
    ----------
    Velocity : device array
        Velocity field.
    EnergyGradient : device array
        Energy gradient field.
    en : device array
        Device buffer storing per site criterion values.
    entmp : device array
        Temporary device buffer used for reduction.
    gridsum_partial : device array
        Partial reduction buffer.
    p_i_d : device array
        Integer parameter array on the device.
    p_i_h : host array
        Integer parameter array on the host.

    Output
    ------
    float
        Sum of the arresting criterion over the full grid.
    """
    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))
    arresting_criteria_kernel[grid3d, block3d](Velocity, EnergyGradient, en, p_i_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[6]
    force = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return force

def do_arrested_newton_flow(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, k1, k2, k3, k4, l1, l2, l3, l4, Temp, en, entmp, gridsum_partial, max_partial, p_i_d, p_f_d, p_i_h, p_f_h, prev_energy, compute_energy, gradient_step_kernel, rk4_kernel, compute_norm=None, do_norm_kernel=None):
    """
    Perform one arrested Newton flow step using RK4 integration.

    Usage
    -----
    Use ``new_energy, err = do_arrested_newton_flow(...)`` to advance the simulation by one minimization step.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Field configuration updated in place.
    d1fd1x : device array
        Buffer for first derivatives.
    d2fd2x : device array
        Buffer for second derivatives.
    EnergyGradient : device array
        Buffer for the energy gradient.
    k1 : device array
        First RK4 position slope.
    k2 : device array
        Second RK4 position slope.
    k3 : device array
        Third RK4 position slope.
    k4 : device array
        Fourth RK4 position slope.
    l1 : device array
        First RK4 velocity slope.
    l2 : device array
        Second RK4 velocity slope.
    l3 : device array
        Third RK4 velocity slope.
    l4 : device array
        Fourth RK4 velocity slope.
    Temp : device array
        Intermediate field buffer.
    en : device array
        Energy buffer.
    entmp : device array
        Temporary energy buffer.
    gridsum_partial : device array
        Partial reduction buffer for energy sums.
    max_partial : device array
        Partial reduction buffer for maximum norms.
    p_i_d : device array
        Integer parameter array on the device.
    p_f_d : device array
        Float parameter array on the device.
    p_i_h : host array
        Integer parameter array on the host.
    p_f_h : host array
        Float parameter array on the host.
    prev_energy : float
        Energy from the previous step.
    compute_energy : function
        Host function that computes the scalar energy.
    gradient_step_kernel : function
        CUDA kernel that computes the gradient step.
    rk4_kernel : function
        CUDA kernel that finalizes the RK4 update.
    compute_norm : function, optional
        Host function that computes a normalization factor.
    do_norm_kernel : function, optional
        CUDA kernel that applies the normalization.

    Output
    ------
    tuple
        Pair ``(new_energy, err)`` containing the updated energy and the maximum norm of the energy gradient.
    """
    grid3d, block3d = launch_3d(p_i_h, threads=(8, 8, 4))

    def normalize_field(target_field):
        """
        Apply the optional field normalization.

        Usage
        -----
        Use ``normalize_field(Temp)`` to normalize an intermediate field buffer when a normalization routine is provided.

        Parameters
        ----------
        target_field : device array
            Field buffer to normalize.

        Output
        ------
        None
            The field is normalized in place when normalization is enabled.
        """
        if (compute_norm is None) or (do_norm_kernel is None):
            return
        norm = compute_norm(target_field, en, entmp, gridsum_partial, p_i_d, p_i_h, p_f_d)
        do_norm_kernel[grid3d, block3d](target_field, norm, p_i_d)
        cuda.synchronize()

    do_rk4_step_kernel[grid3d, block3d](k1, Velocity, Velocity, Temp, Field, Field, 0.0, p_i_d, p_f_d)
    normalize_field(Temp)
    gradient_step_kernel[grid3d, block3d](l1, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)

    do_rk4_step_kernel[grid3d, block3d](k2, Velocity, l1, Temp, Field, k1, 0.5, p_i_d, p_f_d)
    normalize_field(Temp)
    gradient_step_kernel[grid3d, block3d](l2, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)

    do_rk4_step_kernel[grid3d, block3d](k3, Velocity, l2, Temp, Field, k2, 0.5, p_i_d, p_f_d)
    normalize_field(Temp)
    gradient_step_kernel[grid3d, block3d](l3, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)

    do_rk4_step_kernel[grid3d, block3d](k4, Velocity, l3, Temp, Field, k3, 1.0, p_i_d, p_f_d)
    normalize_field(Temp)
    gradient_step_kernel[grid3d, block3d](l4, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)

    rk4_kernel[grid3d, block3d](Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i_d, p_f_d)
    cuda.synchronize()
    normalize_field(Field)

    new_energy = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)
    killkinen = p_i_h[8]

    if killkinen and (new_energy > prev_energy):
        set_field_zero_kernel[grid3d, block3d](Velocity, p_i_d)
        cuda.synchronize()

    err = compute_max_field(EnergyGradient, max_partial, p_i_h)
    return new_energy, err