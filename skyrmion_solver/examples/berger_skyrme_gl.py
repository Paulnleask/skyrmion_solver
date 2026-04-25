"""
To run: python -m skyrmion_solver.examples.berger_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Berger Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,                  # Lattice points
        xsize=10.0, ysize=10.0, zsize=10.0,         # Dimensionless box size
        mpi=1.0, kappa=1.0, alpha=2.0,              # Dimensionless Skyrme parameters
        courant=0.1,                                # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "rational_map", "baryon_number": 4})
    # sim.initialize({"mode": "smorgasbord", "baryon_number": 25, "seed": 7})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()