<p align="center">
  <img src="https://raw.githubusercontent.com/Paulnleask/skyrmion_solver/main/skyrmion_solver/assets/skyrmion_solver.png" width="500">
</p>

<h4 align="center">
CUDA-accelerated finite-difference PDE solver for baryonic solitons in three-dimensional nonlinear field theories, with real-time CUDA–OpenGL volume ray-marching via cuda-python and PyOpenGL.
</h4>

<p align="center">
<a href="https://pypi.org/project/skyrmion_solver/">
  <img src="https://img.shields.io/pypi/v/skyrmion_solver.svg" alt="PyPI">
</a>
<a href="https://github.com/paulnleask/skyrmion_solver/releases">
  <img src="https://img.shields.io/github/v/release/paulnleask/skyrmion_solver?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/paulnleask/skyrmion_solver/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#supported-theories">Supported Theories</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#numerical-method">Numerical method</a> •
  <a href="#license">License</a>
</p>

## About

`skyrmion_solver` is a specialized three-dimensional GPU physics solver for baryons in the low-energy regime of quantum chromodynamics (QCD).
In the large-colour limit, low-energy QCD admits an effective chiral theory of mesons in which baryons arise as topological solitons.
The Skyrme model is one such description: it is an effective Lagrangian involving only the lightest mesons, the pions, in which baryons are realized as non-perturbative excitations of the pion field.

Following on from the release of `soliton_solver`, `skyrmion_solver` has been developed as a dedicated framework for fully three-dimensional Skyrme-type field theories.
It is a Python port of the specialized C++ codebase `cuSkyrmion`, built on the architecture of `soliton_solver`.
It is designed for direct numerical studies of single and multi-Skyrmion configurations, nuclear clustering, crystalline phases, and extensions of the pion-only model involving additional dynamical fields.

From a modelling perspective, the standard Skyrme term is phenomenological, and more natural extensions arise by coupling additional degrees of freedom.
In particular, the omega-meson variant provides a physically motivated stabilization mechanism through a vector field coupled to the baryon current, modifying both the energetics and bulk properties.
Electromagnetic effects can be incorporated via a Maxwell field, introducing Coulomb energy and its backreaction, which becomes increasingly important at larger baryon number.
These extensions lead to fully three-dimensional coupled nonlinear PDE systems that must be solved without symmetry reduction.

The solver is implemented as a GPU-native finite-difference framework using Numba CUDA, where the full field theory is expressed directly as compiled CUDA kernels.
Each lattice site is mapped to a GPU thread, and the nonlinear field equations are evaluated pointwise using high-order finite-difference stencils for spatial derivatives.
Time evolution and energy minimization are implemented through explicit kernel updates, including RK4 integration and accelerated gradient flows such as arrested Newton flow, preserving the structure of the continuum theory while exploiting SIMT execution.
The solver operates at the level of grid-local updates, allowing direct control over multi-field couplings, including omega-meson, rho-meson and Coulomb interactions, without relying on higher-level tensor abstractions.

The architecture follows the same design principle as `soliton_solver`, separating the numerical engine from the physical model via dependency injection.
This enables different Skyrme variants to be explored within a consistent execution model without modifying the core solver.

Visualization is integrated directly into the computational pipeline using CUDA–OpenGL interoperability.
Device buffers are mapped to OpenGL pixel buffer objects (PBOs), allowing CUDA kernels to write RGBA volume data in-place without host transfer.
These volumes are uploaded to 3D textures and rendered via ray-marching, with energy density controlling opacity and field structure encoded in colour.
This design keeps both computation and visualization resident on the GPU, enabling real-time inspection of large-scale three-dimensional simulations of nuclear matter.

---

## Installation

### System requirements

Before installing `skyrmion_solver`, the following system-level dependencies must be available:

- **NVIDIA GPU with CUDA support**
- **CUDA Toolkit** compatible with your GPU
- **OpenGL drivers** compatible with your NVIDIA installation

You can verify your CUDA setup with:

```bash
nvidia-smi
```

and

```bash
nvcc --version
```

### Install from PyPI

```bash
pip install skyrmion-solver
```

### Install from source

```bash
git clone https://github.com/paulnleask/skyrmion_solver.git
cd skyrmion_solver
pip install -e .
```

### Python requirements

- Python 3.10+
- CUDA-capable GPU
- NVIDIA drivers compatible with Numba CUDA

The package also installs the runtime dependencies needed for visualization and GPU interop, including:

- **numba-cuda** for GPU-resident numerical kernels and SIMT-parallel computation
- **glfw** for OpenGL context creation, window management, and input handling
- **PyOpenGL** for Python bindings to the OpenGL rendering pipeline
- **cuda-python** for low-level CUDA driver access and CUDA–OpenGL interoperability

---

## Quickstart

A typical workflow is:

1. Load a theory module.
2. Create a parameter set.
3. Construct a `Simulation`.
4. Initialize the field configuration.
5. Run arrested Newton flow or RK4 time evolution.
6. Visualize the result interactively with the OpenGL viewer.

Typical usage:

```python
from skyrmion_solver.theories import load_theory
from skyrmion_solver.core.simulation import Simulation
theory = load_theory("Nuclear Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=64, ylen=64, zlen=64,              # Lattice points
        xsize=12.0, ysize=12.0, zsize=12.0,     # Dimensionless box size
        mpi=1.0, kappa=1.0,                     # Dimensionless free parameters
        courant=0.2,                            # Time step courant
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "smorgasbord", "baryon_number": 8, "seed": 2})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=3)

if __name__ == "__main__":
    run_gl_simulation()
```

The simulation and rendering pipelines remain on the GPU throughout execution.
Field data is written directly from CUDA kernels into OpenGL buffers using CUDA–OpenGL interoperability, enabling real-time inspection of evolving three-dimensional Skyrmion configurations.

Shown below is a $B=8$ Skyrmion generated using the above typical usage (with `seed=2`) in `skyrmion_solver`.

<p align="center">
  <img src="https://raw.githubusercontent.com/Paulnleask/skyrmion_solver/main/skyrmion_solver/assets/Nuclear_B8.png" width="500">
</p>

---

## Supported Theories

`skyrmion_solver` is currently focused on three-dimensional Skyrme-type theories relevant to nuclear and hadronic physics.

This includes:

- The standard pion-only nuclear Skyrme model
- Omega-meson and rho-meson extensions
- Coulomb or Maxwell-coupled variants
- The lightly bound Skyrme model

All models are GPU-accelerated and compatible with real-time rendering.
New theories can be added by implementing a theory module and registering it with the **theory registry**.
The framework is designed so that additional variants can be introduced without changing the numerical core.
The currently supported theories are shown below:

| Theory | Fields | Lagrangian |
|--------|--------|--------|
| Berger-Skyrme | $U \in \textup{SU}(2)$ | $\mathcal{L}_{\textup{Ber}}\lbrack U\rbrack  = \frac{F_{\pi}^2}{16\hbar}\textup{Tr}(L_{\mu} L^{\mu}) + \frac{\hbar}{32g^2}\textup{Tr}\left(\lbrack L_{\mu},L_{\nu}\rbrack \lbrack L^{\mu},L^{\nu}\rbrack \right) - \frac{F_{\pi}^2m_{\pi}^2}{8\hbar^3}\textup{Tr}(\textup{Id}_2-U) - (\alpha^2-1) \left(\frac{F_{\pi}^2}{8\hbar}L_{\mu}^3L^{3\mu} + \frac{\hbar}{16g^2}\Omega_{\mu\nu}^3\Omega^{3\mu\nu} \right)$ |
| Coulomb-Skyrme | $U \in \textup{SU}(2)$, $A_{0} \in \mathbb{R}$ | $\mathcal{L}_{\textup{Cou}}\lbrack U,A_{0}\rbrack  = \frac{F_{\pi}^2}{16\hbar}\textup{Tr}(L_{\mu} L^{\mu}) + \frac{\hbar}{32g^2} \textup{Tr}\left(\lbrack L_{\mu},L_{\nu}\rbrack \lbrack L^{\mu},L^{\nu}\rbrack \right) - \frac{F_{\pi}^2m_{\pi}^2}{8\hbar^3} \textup{Tr}(\textup{Id}_2-U) + \frac{1}{2\hbar}\|\bm{\nabla}A_{0}\|^2 - \frac{e}{2}A_{0} B^{0}\lbrack U\rbrack $ |
| Lightly Bound Skyrme | $U \in \textup{SU}(2)$ | $\mathcal{L}_{\textup{Lig}}\lbrack U\rbrack  = (1-\alpha)\left(\frac{F_{\pi}^2}{16\hbar}\textup{Tr}(L_{\mu} L^{\mu}) - \frac{F_{\pi}^2m_{\pi}^2}{8\hbar^3}\textup{Tr}(\textup{Id}_2-U) \right) + \frac{\hbar}{32g^2}\textup{Tr}\left(\lbrack L_{\mu},L_{\nu}\rbrack \lbrack L^{\mu},L^{\nu}\rbrack \right) - \frac{\alpha F_{\pi}^4 g^2}{512\hbar^3(1-\alpha)^2} \textup{Tr}(\textup{Id}_2-U)^4$ |
| Nuclear Skyrme | $U \in \textup{SU}(2)$ | $\mathcal{L}_{\textup{Sk}}\lbrack U\rbrack  = \frac{F_{\pi}^2}{16\hbar} \textup{Tr}(L_{\mu} L^{\mu}) + \frac{\hbar}{32g^2} \textup{Tr}\left( \left\lbrack L_{\mu}, L_{\nu}\right\rbrack  \left\lbrack L^{\mu}, L^{\nu} \right\rbrack  \right) - \frac{F_{\pi}^2 m_{\pi}^2}{8\hbar^3} \textup{Tr}(\textup{Id}_2 - U)$ |
| $\omega$-Skyrme | $U \in \textup{SU}(2)$, $\omega \in \mathbb{R}$ | $\mathcal{L}_{\omega}\lbrack U,\omega\rbrack  = \frac{F_{\pi}^2}{16\hbar}\,\textup{Tr}(L_{\mu} L^{\mu}) + \frac{F_{\pi}^2 m_{\pi}^2}{8\hbar^3}\,\textup{Tr}(U-\textup{Id}_2) + \frac{1}{2\hbar}\|\bm{\nabla}\omega\|^2 + \frac{m_{\omega}^2}{2\hbar^3}\omega^2 + \beta_{\omega} \omega B^{0}\lbrack U\rbrack $ |
| $\rho$-Skyrme | $U \in \textup{SU}(2)$, $R_{\mu} \in \mathfrak{su}(2)$ | $\mathcal{L}_{\rho}\lbrack U,R_{\mu}\rbrack  = \frac{F_{\pi}^2}{16\hbar} \textup{Tr}(L_{\mu} L^{\mu}) + \frac{\hbar}{32g^2} \textup{Tr}\left( \lbrack L_{\mu}, L_{\nu}\rbrack  \lbrack L^{\mu}, L^{\nu}\rbrack  \right) -\frac{F_{\pi}^2 m_{\pi}^2}{8\hbar^3} \textup{Tr}\left( \textup{Id}_2 - U \right) -\frac{m_{\rho}^2}{4\hbar^3}\textup{Tr}\left(R_{\mu}^\dagger R^{\mu}\right) - \frac{1}{8\hbar} \textup{Tr}\left(R_{\mu\nu}^\dagger R^{\mu\nu}\right) + \frac{1}{2}\alpha\textup{Tr}\left( R_{\mu\nu}\lbrack L^{\mu}, L^{\nu}\rbrack  \right)$ |

---

## Features

- GPU-native finite-difference PDE solver for three-dimensional nonlinear field theories
- Numba CUDA kernels for pointwise evaluation of field equations
- Explicit RK4 time stepping and accelerated relaxation methods such as arrested Newton flow
- Real-time CUDA–OpenGL visualization with GPU-resident volume rendering
- Dependency-injected architecture separating numerical infrastructure from physical models
- Direct support for multi-field couplings in fully three-dimensional settings
- Interactive viewer for exploring energy density, field structure, and observables
- Export pipeline for simulation data and post-processing

---

## Architecture

```text
skyrmion_solver/
├── core/            GPU numerical engine
├── theories/        modular Skyrme-type field theories
├── visualization/   OpenGL rendering backend
├── examples/        runnable demonstrations
├── version.py
└── pyproject.toml
```

### Core

The numerical core implements the GPU PDE solver infrastructure:

- 4th order central finite-difference operators
- Explicit time-stepping integrators, such as 4th order Runge-Kutta
- Accelerated gradient descent with flow arresting, i.e. Arrested Newton flow
- Simulation driver
- GPU memory management
- Observable reduction utilities

These components are independent of any particular Skyrme variant.

### Theories

Each theory module defines:

- Field content, i.e. the SU(2) Skyrme field and extensions to include vector mesons for example
- Energy functional and its associated variation with respect to the field(s)
- Parameter set, either in physical units or as dimensionless variables
- Initialization routines such as the rational map ansatz or the smorgasbord ansatz
- Physical observables like the charge radius, or moments of inertia tensors
- Optional visualization helpers
- Theory-specific output routines

Theories are introduced via dependency injection, so the execution model stays fixed while the physics changes.

### Visualization

The visualization backend uses CUDA–OpenGL interoperability:

- CUDA writes RGBA volume data directly into mapped buffers
- Volume data is uploaded as 3D textures without host copies
- OpenGL ray-marching renders energy density and field structure in real time

This enables direct interactive inspection of large three-dimensional simulations while minimization or time evolution is running.

---

## Numerical method

Static and metastable Skyrmion configurations are obtained by minimizing a discretized energy functional on a three-dimensional lattice.
Given a discrete energy $E_h[U]$, the solver evolves the fields using explicit GPU kernel updates.
For relaxation, one of the main methods is arrested Newton flow, which introduces a fictitious second-order time evolution,

$$
\ddot{U} = - \nabla_U E_h[U].
$$

The system is advanced using explicit time-stepping schemes such as RK4, while an arrest condition resets velocities when the flow overshoots or begins to climb in energy.
This approach combines the simplicity of explicit local updates with much faster convergence than naive gradient descent for multi-Skyrmion configurations and other stiff nonlinear field problems.
Spatial derivatives are evaluated using high-order finite-difference stencils.
Because each lattice site is mapped to a GPU thread, the solver naturally exploits SIMD/SIMT hardware for fully three-dimensional computations without symmetry reduction.

---

## License

MIT License.
See `LICENSE` for details.