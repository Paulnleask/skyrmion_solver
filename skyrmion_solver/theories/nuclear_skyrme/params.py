"""
Skyrme theory specific parameters, parameter resolution, device packing, and terminal parameter documentation.

Examples
--------
Use ``default_params`` to create a Skyrme model parameter set with optional overrides.
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
    User facing three dimensional Skyrme model parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields in the theory.
    mpi : float, optional
        Pion mass coefficient used in the standard pion mass potential.
    kappa : float, optional
        Coefficient controlling the Skyrme term strength.
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

    Examples
    --------
    Use ``p = Params()`` to create the default Skyrme parameter set.
    Use ``p = Params(mpi=0.5, kappa=1.0, baryon_number=1.0, ansatz="hedgehog")`` to set model parameters and the initial ansatz.
    Use ``rp = p.resolved()`` to derive the resolved parameter set.
    """

    number_total_fields: int = 4

    mpi: float = 1.0
    kappa: float = 1.0

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
    mpi : float
        Pion mass coefficient used in the standard pion mass potential.
    kappa : float
        Skyrme term coefficient.
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

    mpi: float
    kappa: float

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
            mpi=float(p.mpi),
            kappa=float(p.kappa),
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
    Use ``p = default_params(mpi=0.5, baryon_number=2.0, ansatz="rational_map")`` to build a customized Skyrme model parameter set.
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
    The Skyrme kernels assume ``mpi = p_f[8]`` and ``kappa = p_f[9]``.
    The remaining appended entries are reserved for theory specific initialization logic.

    Examples
    --------
    Use ``p_i, p_f = pack_device_params(rp)`` to build the Skyrme device parameter arrays.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        1 if rp.ansatz_hedgehog else 0,
        1 if rp.ansatz_rational_map else 0,
        1 if rp.ansatz_uniform else 0,
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.mpi,
        rp.kappa,
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
    print("  The three dimensional SU(2) Skyrme model uses 4 field components by default.")
    print("  The components are interpreted as sigma, pi1, pi2, and pi3.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  zlen : number of grid points in z-direction")
    print("  xsize : physical grid size in x-direction")
    print("  ysize : physical grid size in y-direction")
    print("  zsize : physical grid size in z-direction")
    print()

    print("Model couplings:")
    print("  mpi : pion mass coefficient in the standard pion mass potential mpi^2 (1 - sigma)")
    print("  kappa : coefficient controlling the Skyrme quartic term")
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
    print("  Theory float parameters are appended in the order: mpi, kappa, baryon_number, isospin_rotation.")
    print("  The Skyrme kernels therefore read mpi from p_f[8] and kappa from p_f[9].")
    print()