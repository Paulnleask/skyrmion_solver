"""
To run: python -m skyrmion_solver.examples.lightly_bound_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Lightly bound Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,                  # Lattice points
        xsize=12.0, ysize=12.0, zsize=12.0,         # Dimensionless box size
        mpi=1.0, alpha=0.95,                        # Dimensionless parameters
        courant=0.15,                               # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "rational_map", "baryon_number": 7})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()