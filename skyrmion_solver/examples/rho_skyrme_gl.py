"""
To run: python -m skyrmion_solver.examples.rho_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Rho-Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,                      # Lattice points
        xsize=2.0, ysize=2.0, zsize=2.0,                # Dimensionless box size
        mass_pion=138.0, f_pi=129.0, e_skyrme=3.65,     # Physical Skyrme parameters
        mass_rho=775.0, alpha=0.0618,                   # Physical rho-meson parameters
        courant=0.05,                                   # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "rational_map", "baryon_number": 4})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()