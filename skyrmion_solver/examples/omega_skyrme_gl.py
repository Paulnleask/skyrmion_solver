"""
To run: python -m skyrmion_solver.examples.omega_skyrme_gl
"""
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Omega-Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,
        xsize=12.0, ysize=12.0, zsize=12.0,
        # f_pi=124.0, m_pi=138.0, m_omega=782.0, beta_omega=15.603,     # mass=0.176, c_omega=98.4
        # f_pi=186.0, m_pi=138.0, m_omega=782.0, beta_omega=8.253,      # mass=0.176, c_omega=34.7
        f_pi=139.8, m_pi=43.91, m_omega=249.5, beta_omega=8.036,      # mass=0.176, c_omega=14.34
        courant=0.2,
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "rational_map", "baryon_number": 3})
    # sim.initialize({"mode": "smorgasbord", "baryon_number": 25, "seed": 7})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()