"""
Nuclear Skyrme theory package registration and user facing theory description.

Examples
--------
Use ``load_theory("Nuclear Skyrme model")`` to load the theory package.
Use ``theory.describe()`` to print a structured summary of the theory.
"""

from __future__ import annotations
from skyrmion_solver.theories.registry import TheorySpec
from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl
from .instructions import print_instructions

THEORY_SPEC = TheorySpec(
    name="Nuclear Skyrme model",
    aliases=("Skyrme Model", "3D skyrmions", "Skyrmions"),
    import_path="skyrmion_solver.theories.nuclear_skyrme",
    description="Skyrmions in the 3D nuclear Skyrme model, with the standard pion mass potential.",
    version="1.0"
)

def _print_section(title: str) -> None:
    """
    Print a section heading.

    Parameters
    ----------
    title : str
        Section title to print.

    Returns
    -------
    None
        The section heading is printed to the terminal.

    Examples
    --------
    Use ``_print_section("Aliases")`` to print a section heading.
    """
    print(title)
    print("-" * len(title))

def _print_metadata() -> None:
    """
    Print the registry metadata for the nuclear Skyrme theory.

    Returns
    -------
    None
        The theory metadata are printed to the terminal.

    Examples
    --------
    Use ``_print_metadata()`` to print the theory metadata.
    """
    print("=" * 72)
    print(f"{THEORY_SPEC.name} (version {THEORY_SPEC.version})")
    print("=" * 72)
    print(THEORY_SPEC.description)
    print()
    print(f"Import path: {THEORY_SPEC.import_path}")
    print()

    _print_section("Aliases")

    if THEORY_SPEC.aliases:
        print(", ".join(THEORY_SPEC.aliases))
    else:
        print("None")

    print()

def _print_submodules() -> None:
    """
    Print the main nuclear Skyrme submodules.

    Returns
    -------
    None
        The submodule names are printed to the terminal.

    Examples
    --------
    Use ``_print_submodules()`` to print the main submodules.
    """
    _print_section("Main submodules")
    print("params")
    print("kernels")
    print("initial_config")
    print("observables")
    print("io")
    print("render_gl")
    print()

def _print_parameter_information() -> None:
    """
    Print parameter information for the nuclear Skyrme theory.

    Returns
    -------
    None
        The parameter information is printed to the terminal.

    Examples
    --------
    Use ``_print_parameter_information()`` to print parameter details.
    """
    _print_section("Parameter information")

    describe_fn = getattr(params, "describe", None)

    if callable(describe_fn):
        describe_fn()
    else:
        print("No detailed parameter description is defined.")

    print()

def _print_notes() -> None:
    """
    Print a short summary of the nuclear Skyrme theory package.

    Returns
    -------
    None
        The notes are printed to the terminal.

    Examples
    --------
    Use ``_print_notes()`` to print a short package summary.
    """
    _print_section("Notes")
    print("This theory package provides the nuclear Skyrme model in two spatial dimensions.")
    print("It includes kernels for the field equations, initial condition utilities, observable calculations, I/O helpers, and OpenGL rendering support.")
    print("The model supports multiple potential choices through the parameter layer.")
    print()

def _print_instructions() -> None:
    """
    Print additional usage instructions for the nuclear Skyrme theory.

    Returns
    -------
    None
        The instructions are printed to the terminal.

    Examples
    --------
    Use ``_print_instructions()`` to print additional usage instructions.
    """
    _print_section("Instructions")
    print_instructions()
    print()

def describe() -> None:
    """
    Print a structured description of the nuclear Skyrme theory.

    Returns
    -------
    None
        The theory description is printed to the terminal.

    Examples
    --------
    Use ``theory.describe()`` to print the nuclear Skyrme theory description.
    """
    _print_metadata()
    _print_submodules()
    _print_notes()
    _print_parameter_information()
    _print_instructions()