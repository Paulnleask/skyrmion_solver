"""
Omega-Skyrme theory specific parameters, parameter resolution, device packing, and terminal parameter documentation.

Examples
--------
Use ``default_params`` to create an omega-Skyrme model parameter set with optional overrides.
Use ``Params.resolved()`` to derive the fully resolved theory parameters.
Use ``pack_device_params`` to build the device parameter arrays for the Skyrme kernels.
Use ``describe()`` to print the Skyrme parameter documentation.
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
    User facing three dimensional omega-Skyrme model parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields in the theory.
    number_skyrme_fields : int, optional
        Number of Skyrme field components.
    f_pi : float, optional
        Pion decay constant used for parameter bookkeeping.
    m_pi : float, optional
        Physical pion mass used for parameter bookkeeping.
    m_omega : float, optional
        Physical omega-meson mass used for parameter bookkeeping.
    beta_omega : float, optional
        Dimensionless omega coupling used for parameter bookkeeping.
    mass : float
        Dimensionless pion mass coefficient used in the kernels.
    c_omega : float
        Dimensionless omega coupling used in the kernels.
    baryon_number : float, optional
        Target baryon number used by the initial ansatz.
    isospin_rotation : float, optional
        Internal rotation angle used by the initial ansatz.
    ansatz : str, optional
        Initial ansatz type.

    Returns
    -------
    None
        The dataclass stores the Skyrme model parameter set.

    Notes
    -----
    The kernel-facing dimensionless couplings are derived at resolve time as
    ``mass = m_pi / m_omega`` and ``c_omega = m_omega * beta_omega / f_pi``.

    Examples
    --------
    Use ``p = Params()`` to create the default Skyrme parameter set.
    Use ``p = Params(f_pi=186.0, m_pi=138.0, m_omega=782.0, beta_omega=8.253)`` to set the physical bookkeeping parameters.
    Use ``rp = p.resolved()`` to derive the resolved parameter set.
    """

    number_total_fields: int = 5
    number_skyrme_fields: int = 4

    f_pi: float = 186.0
    m_pi: float = 138.0
    m_omega: float = 782.0
    beta_omega: float = 8.253

    mass: float | None = None
    c_omega: float | None = None

    baryon_number: float = 1.0
    isospin_rotation: float = 0.0
    ansatz: str = "hedgehog"

    def resolved(self) -> "ResolvedParams":
        """
        Convert user facing parameters into resolved Skyrme model parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved Skyrme model parameters.

        Examples
        --------
        Use ``rp = p.resolved()`` to derive the resolved Skyrme model parameters.
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved three dimensional Skyrme model parameters.

    Parameters
    ----------
    number_skyrme_fields : int
        Number of Skyrme field components.
    f_pi : float
        Pion decay constant used for parameter bookkeeping.
    m_pi : float
        Physical pion mass used for parameter bookkeeping.
    m_omega : float
        Physical omega-meson mass used for parameter bookkeeping.
    beta_omega : float
        Dimensionless omega coupling used for parameter bookkeeping.
    mass : float
        Dimensionless pion mass coefficient used in the kernels.
    c_omega : float
        Dimensionless omega coupling used in the kernels.
    baryon_number : float
        Target baryon number used in the initial ansatz.
    isospin_rotation : float
        Internal rotation angle used in the initial ansatz.
    ansatz_hedgehog : bool
        Whether the hedgehog ansatz is enabled.
    ansatz_rational_map : bool
        Whether the rational map ansatz is enabled.
    ansatz_uniform : bool
        Whether the uniform ansatz is enabled.

    Returns
    -------
    None
        The dataclass stores the resolved Skyrme model parameter set.

    Examples
    --------
    Use ``rp = ResolvedParams.from_params(p)`` to build the resolved Skyrme model parameters from ``p``.
    """

    number_skyrme_fields: int

    f_pi: float
    m_pi: float
    m_omega: float
    beta_omega: float

    mass: float
    c_omega: float

    baryon_number: float
    isospin_rotation: float

    ansatz_hedgehog: bool
    ansatz_rational_map: bool
    ansatz_uniform: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved Skyrme model parameters from user facing parameters.

        Parameters
        ----------
        p : Params
            User facing Skyrme model parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved Skyrme model parameters.

        Raises
        ------
        ValueError
            Raised if an unknown ansatz name is supplied.
        ValueError
            Raised if ``f_pi`` or ``m_omega`` is zero.

        Examples
        --------
        Use ``rp = ResolvedParams.from_params(p)`` to resolve the Skyrme model parameters.
        """
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "hedgehog").lower().strip()
        ansatz_hedgehog = ans == "hedgehog"
        ansatz_rational_map = ans in ("rational_map", "rational map", "rational-map")
        ansatz_uniform = ans == "uniform"

        if not (ansatz_hedgehog or ansatz_rational_map or ansatz_uniform):
            raise ValueError("Unknown ansatz '{0}'. Valid: hedgehog, rational_map, uniform".format(p.ansatz))

        if float(p.m_omega) == 0.0:
            raise ValueError("m_omega must be nonzero when deriving mass = m_pi / m_omega.")
        if float(p.f_pi) == 0.0:
            raise ValueError("f_pi must be nonzero when deriving c_omega = m_omega * beta_omega / f_pi.")

        if p.mass is None:
            mass = float(p.m_pi) / float(p.m_omega)
        else:
            mass = p.mass
        if p.c_omega is None:
            c_omega = float(p.m_omega) * float(p.beta_omega) / float(p.f_pi)
        else:
            c_omega = p.c_omega

        return ResolvedParams(
            xlen=core.xlen,
            ylen=core.ylen,
            zlen=core.zlen,
            halo=core.halo,
            number_coordinates=core.number_coordinates,
            number_total_fields=core.number_total_fields,
            number_skyrme_fields=int(p.number_skyrme_fields),
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
            f_pi=float(p.f_pi),
            m_pi=float(p.m_pi),
            m_omega=float(p.m_omega),
            beta_omega=float(p.beta_omega),
            mass=mass,
            c_omega=c_omega,
            baryon_number=float(p.baryon_number),
            isospin_rotation=float(p.isospin_rotation),
            ansatz_hedgehog=ansatz_hedgehog,
            ansatz_rational_map=ansatz_rational_map,
            ansatz_uniform=ansatz_uniform,
        )


def default_params(**overrides) -> Params:
    """
    Construct Skyrme model parameters using defaults plus user overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to ``Params.with_()``.

    Returns
    -------
    Params
        Parameter object with the requested overrides applied.

    Examples
    --------
    Use ``p = default_params(f_pi=186.0, m_pi=138.0, m_omega=782.0, beta_omega=8.253)`` to build a customized Skyrme model parameter set.
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved Skyrme model parameters into device ABI arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved Skyrme model parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple ``(p_i, p_f)`` containing the integer and floating point device parameter arrays.

    Notes
    -----
    The resulting float array appends the theory parameters after the core array.
    The Skyrme kernels assume ``mass = p_f[8]`` and ``c_omega = p_f[9]``.
    The physical bookkeeping parameters ``f_pi``, ``m_pi``, ``m_omega``, and ``beta_omega`` are not packed to the device because the current kernels do not use them.

    Examples
    --------
    Use ``p_i, p_f = pack_device_params(rp)`` to build the Skyrme device parameter arrays.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        rp.number_skyrme_fields,
        1 if rp.ansatz_hedgehog else 0,
        1 if rp.ansatz_rational_map else 0,
        1 if rp.ansatz_uniform else 0,
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.mass,
        rp.c_omega,
        rp.baryon_number,
        rp.isospin_rotation,
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))

    return p_i, p_f


def describe() -> None:
    """
    Print a readable description of the Skyrme model parameter set.

    Returns
    -------
    None
        The parameter information is printed to the terminal.

    Examples
    --------
    Use ``describe()`` to print the Skyrme model parameter documentation.
    """
    print("Field content:")
    print("  The three dimensional omega-Skyrme model uses 5 field components by default.")
    print("  The components are interpreted as sigma, pi1, pi2, pi3, and omega.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  zlen : number of grid points in z-direction")
    print("  xsize : physical grid size in x-direction")
    print("  ysize : physical grid size in y-direction")
    print("  zsize : physical grid size in z-direction")
    print()

    print("Physical bookkeeping parameters:")
    print("  f_pi : pion decay constant")
    print("  m_pi : physical pion mass")
    print("  m_omega : physical omega-meson mass")
    print("  beta_omega : omega coupling parameter")
    print()

    print("Derived kernel couplings:")
    print("  mass = m_pi / m_omega")
    print("  c_omega = m_omega * beta_omega / f_pi")
    print()

    print("Initial condition controls:")
    print("  baryon_number : target baryon number used by the initial ansatz")
    print("  isospin_rotation : internal rotation angle used by the ansatz")
    print("  mode : initial-condition type")
    print("  Supported mode types:")
    print("    hedgehog")
    print("    rational_map")
    print("    uniform")
    print("    smorgasbord")
    print()

    print("Device packing notes:")
    print("  Core parameters are packed first using skyrmion_solver.core.params.pack_device_params.")
    print("  Theory integer parameters are appended in the order: number_skyrme_fields, hedgehog, rational_map, uniform.")
    print("  Theory float parameters are appended in the order: mass, c_omega, baryon_number, isospin_rotation.")
    print("  The Skyrme kernels therefore read mass from p_f[8] and c_omega from p_f[9].")
    print("  The bookkeeping parameters f_pi, m_pi, m_omega, and beta_omega are stored in Params/ResolvedParams but are not packed to the device.")
    print()