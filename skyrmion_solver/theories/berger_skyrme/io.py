"""
Berger Skyrme theory specific output bundle.

Examples
--------
Use ``output_data_bundle`` to export the Berger Skyrme simulation outputs for plotting.
"""

from __future__ import annotations
import numpy as np
from skyrmion_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_BaryonDensity: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    zlen: int,
    number_coordinates: int = 3,
    number_total_fields: int = 4,
    precision: int = 32,
) -> None:
    """
    Write the Berger Skyrme output data bundle.

    Parameters
    ----------
    output_dir : str
        Directory where the output files will be written.
    h_Field : ndarray
        Flattened host array containing the Skyrme field components
        ``(sigma, pi1, pi2, pi3)``.
    h_EnergyDensity : ndarray
        Flattened host array containing the energy density.
    h_BaryonDensity : ndarray
        Flattened host array containing the baryon density.
    h_grid : ndarray
        Flattened host coordinate grid.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.
    number_coordinates : int, optional
        Number of coordinate components stored in the grid.
    number_total_fields : int, optional
        Number of field components stored in the field array.
    precision : int, optional
        Number of significant digits used in the output format.

    Returns
    -------
    None
        The requested output files are written to the output directory.

    Examples
    --------
    Use ``output_data_bundle(output_dir, h_Field, h_EnergyDensity, h_BaryonDensity, h_grid, xlen, ylen, zlen)`` to write the simulation output bundle.
    """
    arrays = {
        "grid": h_grid,
        "Field": h_Field,
        "EnergyDensity": h_EnergyDensity,
        "BaryonDensity": h_BaryonDensity,
    }

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("grid", 2, "zGrid.dat"),
        ("Field", 0, "SigmaField.dat"),
        ("Field", 1, "PionField1.dat"),
        ("Field", 2, "PionField2.dat"),
        ("Field", 3, "PionField3.dat"),
        ("BaryonDensity", 0, "BaryonDensity.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
    ]

    output_data_bundle_core(
        output_dir,
        h_Field=h_Field,
        h_grid=h_grid,
        xlen=xlen,
        ylen=ylen,
        zlen=zlen,
        number_coordinates=number_coordinates,
        number_total_fields=number_total_fields,
        precision=precision,
        bundle_spec=bundle_spec,
        arrays=arrays,
        lattice_points_name="LatticePoints.dat",
        lattice_vectors_name="LatticeVectors.dat",
        field_dump_name="Field.dat",
    )