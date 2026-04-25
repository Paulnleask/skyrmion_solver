"""
Rho-Skyrme theory specific parameters, parameter resolution, device packing, and terminal parameter documentation.

Examples
--------
Use ``default_params`` to create a Rho-Skyrme model set with optional overrides.
Use ``Params.resolved()`` to derive the fully resolved theory parameters.
Use ``pack_device_params`` to build the device parameter arrays for the rho-Skyrme kernels.
Use ``describe()`` to print the rho-Skyrme parameter documentation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from skyrmion_solver.core.params import Params as CoreParams
from skyrmion_solver.core.params import ResolvedParams as CoreResolvedParams
from skyrmion_solver.core.params import pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    User facing three dimensional rho-Skyrme model parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields in the theory.
        The default layout is 4 Skyrme fields plus 9 rho-meson fields.
    mass_pion : float, optional
        Physical pion mass in MeV.
    mass_rho : float, optional
        Physical rho-meson mass in MeV.
    f_pi : float, optional
        Physical pion decay constant in MeV.
    e_skyrme : float, optional
        Dimensionless Skyrme parameter used in the paper's rescaling.
    alpha : float, optional
        Physical rho-pion coupling parameter.
    mpi : float or None, optional
        Dimensionless pion mass coefficient used by the kernels.
        If ``None``, it is derived from the physical inputs.
    mrho : float or None, optional
        Dimensionless rho-meson mass coefficient used by the kernels.
        If ``None``, it is derived from the physical inputs.
    c_alpha : float or None, optional
        Dimensionless rho-pion coupling used by the kernels.
        If ``None``, it is derived from the physical inputs.
    baryon_number : float, optional
        Target baryon number used by the initial ansatz.
    isospin_rotation : float, optional
        Internal rotation angle used by the initial ansatz.
    ansatz : str, optional
        Initial ansatz type.

    Notes
    -----
    The kernels use the dimensionless parameters ``mpi``, ``mrho``, and ``c_alpha``.
    By default these are derived from the physical inputs using the paper's rescaling,
    but each may be overridden explicitly.
    """

    number_total_fields: int = 13

    mass_pion: float = 138.0
    mass_rho: float = 775.0
    f_pi: float = 129.0
    e_skyrme: float = 1.0
    alpha: float = 0.0617

    mpi: float | None = None
    mrho: float | None = None
    c_alpha: float | None = None

    baryon_number: float = 1.0
    isospin_rotation: float = 0.0
    ansatz: str = "hedgehog"

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved three dimensional rho-Skyrme model parameters.

    Parameters
    ----------
    mass_pion : float
        Physical pion mass in MeV.
    mass_rho : float
        Physical rho-meson mass in MeV.
    f_pi : float
        Physical pion decay constant in MeV.
    e_skyrme : float
        Dimensionless Skyrme parameter used in the rescaling.
    alpha : float
        Physical rho-pion coupling parameter.
    mpi : float
        Dimensionless pion mass coefficient used by the kernels.
    mrho : float
        Dimensionless rho-meson mass coefficient used by the kernels.
    c_alpha : float
        Dimensionless rho-pion coupling used by the kernels.
    mpi_is_derived : bool
        Whether ``mpi`` was derived from the physical inputs.
    mrho_is_derived : bool
        Whether ``mrho`` was derived from the physical inputs.
    c_alpha_is_derived : bool
        Whether ``c_alpha`` was derived from the physical inputs.
    baryon_number : float
        Target baryon number used in the initial ansatz.
    isospin_rotation : float
        Internal rotation angle used in the initial ansatz.
    ansatz_hedgehog : bool
        Whether the hedgehog ansatz is enabled.
    ansatz_rational_map : bool
        Whether the rational-map ansatz is enabled.
    ansatz_uniform : bool
        Whether the uniform ansatz is enabled.
    """

    mass_pion: float
    mass_rho: float
    f_pi: float
    e_skyrme: float
    alpha: float

    mpi: float
    mrho: float
    c_alpha: float

    mpi_is_derived: bool
    mrho_is_derived: bool
    c_alpha_is_derived: bool

    baryon_number: float
    isospin_rotation: float

    ansatz_hedgehog: bool
    ansatz_rational_map: bool
    ansatz_uniform: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved rho-Skyrme model parameters from user facing parameters.

        Parameters
        ----------
        p : Params
            User facing rho-Skyrme model parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved rho-Skyrme model parameters.

        Raises
        ------
        ValueError
            Raised if an unknown ansatz name is supplied.
            Raised if ``e_skyrme`` or ``f_pi`` is zero and a derived quantity is required.
        """
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "hedgehog").lower().strip()
        ansatz_hedgehog = ans == "hedgehog"
        ansatz_rational_map = ans in ("rational_map", "rational map", "rational-map")
        ansatz_uniform = ans == "uniform"

        if not (ansatz_hedgehog or ansatz_rational_map or ansatz_uniform):
            raise ValueError("Unknown ansatz '{0}'. Valid: hedgehog, rational_map, uniform".format(p.ansatz))

        need_rescaling = (p.mpi is None) or (p.mrho is None) or (p.c_alpha is None)
        if need_rescaling and float(p.e_skyrme) == 0.0:
            raise ValueError("e_skyrme must be nonzero when deriving mpi, mrho, or c_alpha.")
        if need_rescaling and float(p.f_pi) == 0.0:
            raise ValueError("f_pi must be nonzero when deriving mpi or mrho.")

        denom = float(p.e_skyrme) * float(p.f_pi)

        mpi_is_derived = p.mpi is None
        mrho_is_derived = p.mrho is None
        c_alpha_is_derived = p.c_alpha is None

        mpi = 2.0 * float(p.mass_pion) / denom if mpi_is_derived else float(p.mpi)
        mrho = 2.0 * float(p.mass_rho) / denom if mrho_is_derived else float(p.mrho)
        c_alpha = float(p.alpha) * float(p.e_skyrme) if c_alpha_is_derived else float(p.c_alpha)

        return ResolvedParams(
            xlen=core.xlen,
            ylen=core.ylen,
            zlen=core.zlen,
            halo=core.halo,
            number_coordinates=core.number_coordinates,
            number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid,
            dim_fields=core.dim_fields,
            killkinen=core.killkinen,
            newtonflow=core.newtonflow,
            unit_length=core.unit_length,
            xsize=core.xsize,
            ysize=core.ysize,
            zsize=core.zsize,
            lsx=core.lsx,
            lsy=core.lsy,
            lsz=core.lsz,
            grid_volume=core.grid_volume,
            time_step=core.time_step,
            mass_pion=float(p.mass_pion),
            mass_rho=float(p.mass_rho),
            f_pi=float(p.f_pi),
            e_skyrme=float(p.e_skyrme),
            alpha=float(p.alpha),
            mpi=mpi,
            mrho=mrho,
            c_alpha=c_alpha,
            mpi_is_derived=mpi_is_derived,
            mrho_is_derived=mrho_is_derived,
            c_alpha_is_derived=c_alpha_is_derived,
            baryon_number=float(p.baryon_number),
            isospin_rotation=float(p.isospin_rotation),
            ansatz_hedgehog=ansatz_hedgehog,
            ansatz_rational_map=ansatz_rational_map,
            ansatz_uniform=ansatz_uniform,
        )


def default_params(**overrides) -> Params:
    """
    Create a default rho-Skyrme parameter set with optional overrides.

    Parameters
    ----------
    **overrides
        Keyword overrides passed into ``Params.with_``.

    Returns
    -------
    Params
        Parameter set with requested overrides applied.
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved rho-Skyrme model parameters into device ABI arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved rho-Skyrme model parameters.

    Returns
    -------
    tuple[ndarray, ndarray]
        Integer and float device parameter arrays.

    Notes
    -----
    Only parameters actually used by the kernels are packed.
    The physical inputs ``mass_pion``, ``mass_rho``, ``f_pi``, ``e_skyrme``, and ``alpha``
    are not packed because the kernels do not read them directly.
    The theory float parameters are appended in the order:
    ``mpi, mrho, c_alpha, baryon_number, isospin_rotation``.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        1 if rp.ansatz_hedgehog else 0,
        1 if rp.ansatz_rational_map else 0,
        1 if rp.ansatz_uniform else 0,
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.mpi,
        rp.mrho,
        rp.c_alpha,
        rp.baryon_number,
        rp.isospin_rotation,
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f


def describe() -> None:
    """
    Print documentation for the rho-Skyrme parameter set.

    Returns
    -------
    None
        The parameter documentation is printed to the terminal.
    """
    print("Field content:")
    print("  The three dimensional rho-Skyrme model uses 13 field components by default.")
    print("  The components are interpreted as:")
    print("    0..3   : sigma, pi1, pi2, pi3")
    print("    4..12  : rho_i^a laid out as 4 + 3*i + a")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  zlen : number of grid points in z-direction")
    print("  xsize : physical grid size in x-direction")
    print("  ysize : physical grid size in y-direction")
    print("  zsize : physical grid size in z-direction")
    print()

    print("Physical model inputs:")
    print("  mass_pion : physical pion mass in MeV")
    print("  mass_rho  : physical rho-meson mass in MeV")
    print("  f_pi      : physical pion decay constant in MeV")
    print("  e_skyrme  : dimensionless Skyrme parameter used in the rescaling")
    print("  alpha     : physical rho-pion coupling parameter")
    print()

    print("Derived kernel parameters:")
    print("  mpi     = 2 * mass_pion / (e_skyrme * f_pi), unless overridden")
    print("  mrho    = 2 * mass_rho / (e_skyrme * f_pi), unless overridden")
    print("  c_alpha = alpha * e_skyrme, unless overridden")
    print()

    print("Initial condition controls:")
    print("  baryon_number : target baryon number used by the initial ansatz")
    print("  isospin_rotation : internal rotation angle used by the ansatz")
    print("  ansatz : initial-condition type")
    print("  Supported ansatz values:")
    print("    hedgehog")
    print("    rational_map")
    print("    uniform")
    print()

    print("Device packing notes:")
    print("  Core parameters are packed first using skyrmion_solver.core.params.pack_device_params.")
    print("  The physical inputs are not packed because the kernels do not read them directly.")
    print("  Only the kernel parameters actually used by the theory kernels are appended.")
    print("  Theory float parameters are appended in the order:")
    print("    mpi, mrho, c_alpha, baryon_number, isospin_rotation")
    print()
    print("  With this layout, the kernels read:")
    print("    mpi              from p_f[8]")
    print("    mrho             from p_f[9]")
    print("    c_alpha          from p_f[10]")
    print("    baryon_number    from p_f[11]")
    print("    isospin_rotation from p_f[12]")