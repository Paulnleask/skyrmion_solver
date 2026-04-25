"""
To run: python -m skyrmion_solver.examples.nuclear_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Nuclear Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,                  # Lattice points
        xsize=12.0, ysize=12.0, zsize=12.0,         # Dimensionless box size
        mpi=1.0, kappa=1.0,                         # Dimensionless parameters
        courant=0.2,                                # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "smorgasbord", "baryon_number": 8, "seed": 2})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()