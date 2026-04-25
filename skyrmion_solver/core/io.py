"""
Host side I/O utilities for writing simulation outputs as plain text .dat files on a three dimensional lattice.

Usage
-----
Use ``output_field_dat`` to write a full multi component field to disk.
Use ``output_iteration_data_dat`` to write scalar iteration data to disk.
Use ``output_density_data_dat`` to write a single component grid to disk.
Use ``output_data_bundle_core`` to write a complete output bundle for plotting.

Output
------
This module writes lattice metadata, flattened field data, and optional density style outputs for three dimensional simulations.
"""

from __future__ import annotations
import os
import numpy as np

def _flat(a: int, x: int, y: int, z: int, xlen: int, ylen: int, zlen: int) -> int:
    """
    Compute the flattened index for a component and lattice site.

    Usage
    -----
    Use ``i = _flat(a, x, y, z, xlen, ylen, zlen)`` to access a flattened field or grid array.

    Parameters
    ----------
    a : int
        Component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    z : int
        Lattice index along the z direction.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.

    Output
    ------
    int
        Flattened index into the one dimensional array.
    """
    return z + y * zlen + x * ylen * zlen + a * xlen * ylen * zlen

def output_field_dat(field_flat: np.ndarray, path: str, xlen: int, ylen: int, zlen: int, nfields: int, precision: int = 32) -> None:
    """
    Write a full multi component field to a plain text .dat file.

    Usage
    -----
    Use ``output_field_dat(h_Field, "out/d_Field.dat", xlen, ylen, zlen, nfields, precision=32)`` to write the full field.

    Parameters
    ----------
    field_flat : ndarray
        Flattened field array containing all components and lattice sites.
    path : str
        Output file path.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.
    nfields : int
        Number of field components to write.
    precision : int, optional
        Number of significant digits used in the output format.

    Output
    ------
    None
        The field data are written to ``path``.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for a in range(nfields):
            for x in range(xlen):
                for y in range(ylen):
                    for z in range(zlen):
                        f.write(fmt.format(float(field_flat[_flat(a, x, y, z, xlen, ylen, zlen)])))
                        f.write("\t")
                    f.write("\n")
                f.write("\n")
            f.write("\n")

def output_iteration_data_dat(values, path: str, precision: int = 32) -> None:
    """
    Write scalar values as a single tab separated line in a plain text .dat file.

    Usage
    -----
    Use ``output_iteration_data_dat([energy, err], "out/IterationData.dat", precision=16)`` to write iteration data.

    Parameters
    ----------
    values : iterable
        Scalar values to write.
    path : str
        Output file path.
    precision : int, optional
        Number of significant digits used in the output format.

    Output
    ------
    None
        The values are written to ``path``.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for v in values:
            f.write(fmt.format(float(v)))
            f.write("\t")

def output_density_data_dat(density_flat: np.ndarray, a: int, path: str, xlen: int, ylen: int, zlen: int, precision: int = 32) -> None:
    """
    Write one component of a flattened grid to a plain text .dat file.

    Usage
    -----
    Use ``output_density_data_dat(density_flat, 0, "out/Density0.dat", xlen, ylen, zlen, precision=32)`` to write one component grid.

    Parameters
    ----------
    density_flat : ndarray
        Flattened array containing one or more component grids.
    a : int
        Component index to write.
    path : str
        Output file path.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.
    precision : int, optional
        Number of significant digits used in the output format.

    Output
    ------
    None
        The selected component grid is written to ``path``.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for x in range(xlen):
            for y in range(ylen):
                for z in range(zlen):
                    f.write(fmt.format(float(density_flat[_flat(a, x, y, z, xlen, ylen, zlen)])))
                    f.write("\t")
                f.write("\n")
            f.write("\n")

def output_data_bundle_core(
    output_dir: str,
    *,
    h_Field: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    zlen: int,
    number_coordinates: int,
    number_total_fields: int,
    precision: int,
    bundle_spec: list[tuple[str, int, str]],
    arrays: dict[str, np.ndarray],
    lattice_points_name: str = "LatticePoints.dat",
    lattice_vectors_name: str = "LatticeVectors.dat",
    field_dump_name: str = "d_Field.dat",
) -> None:
    """
    Write lattice metadata, a full field dump, and optional component grids.

    Usage
    -----
    Use ``output_data_bundle_core(output_dir, h_Field=h_Field, h_grid=h_grid, xlen=xlen, ylen=ylen, zlen=zlen, number_coordinates=3, number_total_fields=nfields, precision=32, bundle_spec=[("Density", 0, "Density0.dat")], arrays={"Density": density_flat})`` to write a complete output bundle.

    Parameters
    ----------
    output_dir : str
        Directory used for all output files.
    h_Field : ndarray
        Flattened host field array.
    h_grid : ndarray
        Flattened host coordinate grid.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.
    number_coordinates : int
        Number of coordinate components stored in ``h_grid``.
    number_total_fields : int
        Number of field components stored in ``h_Field``.
    precision : int
        Number of significant digits used in the output format.
    bundle_spec : list of tuple
        List of ``(array_key, component_index, filename)`` entries describing optional outputs.
    arrays : dict of ndarray
        Mapping from array names to flattened arrays used for optional outputs.
    lattice_points_name : str, optional
        Filename for the lattice point counts.
    lattice_vectors_name : str, optional
        Filename for the lattice vectors.
    field_dump_name : str, optional
        Filename for the full field dump.

    Output
    ------
    None
        The output directory and requested files are written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    lattice_points = np.array([xlen, ylen, zlen], dtype=np.float64)

    x_max = float(h_grid[_flat(0, xlen - 1, ylen - 1, zlen - 1, xlen, ylen, zlen)]) if number_coordinates >= 1 else 0.0
    y_max = float(h_grid[_flat(1, xlen - 1, ylen - 1, zlen - 1, xlen, ylen, zlen)]) if number_coordinates >= 2 else 0.0
    z_max = float(h_grid[_flat(2, xlen - 1, ylen - 1, zlen - 1, xlen, ylen, zlen)]) if number_coordinates >= 3 else 0.0

    lattice_vectors = np.array([
        x_max, 0.0, 0.0,
        0.0, y_max, 0.0,
        0.0, 0.0, z_max,
    ], dtype=np.float64)

    output_iteration_data_dat(lattice_points[:number_coordinates], os.path.join(output_dir, lattice_points_name), precision=precision)
    output_iteration_data_dat(lattice_vectors[:number_coordinates * number_coordinates], os.path.join(output_dir, lattice_vectors_name), precision=precision)

    output_field_dat(h_Field, os.path.join(output_dir, field_dump_name), xlen, ylen, zlen, number_total_fields, precision=precision)

    for array_key, a, filename in bundle_spec:
        if array_key not in arrays:
            continue
        arr = arrays[array_key]
        if arr is None:
            continue
        output_density_data_dat(arr, int(a), os.path.join(output_dir, filename), xlen, ylen, zlen, precision=precision)