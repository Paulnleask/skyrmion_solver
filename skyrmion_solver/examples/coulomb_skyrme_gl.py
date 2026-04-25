"""
To run: python -m skyrmion_solver.examples.coulomb_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Coulomb-Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,                          # Lattice points
        xsize=10.0, ysize=10.0, zsize=10.0,                 # Dimensionless box size
        f_pi=105.9, m_pi=138.0, g=4.010, e=0.302822,        # Physical parameters
        # mass=0.65, kappa=0.737,                           # Dimensionless parameters
        courant=0.2,                                        # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "rational_map", "baryon_number": 4})
    # sim.initialize({"mode": "smorgasbord", "baryon_number": 12, "seed": 1})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()