# JAX-EP: Differentiable Cardiac Electrophysiology Framework
JAX-EP is a fully differentiable cardiac EP solver and parameter-inversion framework built in JAX [1], enabling GPU-accelerated monodomain simulations and gradient-based inference of personalised tissue and cellular EP parameters. This folder contains the JAX-EP parameter-learning pipeline: recovering five cell-kinetics and conductivity parameters from simulated omnipolar electrograms (EGMs). The codebase progresses systematically from cellular-level validation through full left-atrial (LA) forward simulation to parameter learning on a full 3D LA manifold.

## About JAX-EP

This README communicates structural, mathematical, and execution information about the JAX-EP project. It serves as the primary technical documentation for repository visitors and contributors.

The repository is organized into four interconnected core components:

* **0D Cellular Validation**: Verification of the JAX-based modified Mitchell-Schaeffer (mMS) [1] single-cell EP formulation  against an independent openCARP reference trace to ensure core algorithmic and differentiability consistency.
* **2D Spatial Validation**: Combined evaluation of the discrete spatial diffusion solver alongside localized ionic kinetics, benchmarked directly via conduction velocity profiles on an idealized 2D flat plate geometry.
* **Fully Differentiable Monodomain Solver**: Anisotropic Finite Element Method (FEM) propagation mechanics compiled via `jax.jit` and accelerated on modern GPU architectures, showcased through high-performance Left Atrial (LA) forward simulations.
* **Parameter Inversion Pipeline**: Benefiting from the differentiability of the JAX-EP monodomain solver, a dual-phase optimization pipeline combining multi-stage Nelder–Mead optimization with localized finite-difference L-BFGS-B refinement was developed to recover five cellular-kinetic and conductivity parameters from clinically motivated omnipolar electrograms (EGMs).


---
<img src="parameter_learning/Forward_mode_autodiff.png" alt="JAX-EP Forward-mode Automatic Differentiation Validation" width="650">


JAX-EP provides a scalable computational framework for solving monodomain equations while exposing internal parameters to optimization frameworks. It achieves:
1. **Accelerated Forward Models**: Over 50x GPU-to-CPU simulation speedups using Custom Crank-Nicolson solvers and `jax.jit`.
2. **Stable Gradients**: Leverages forward-mode automatic differentiation (`jax.jacfwd`) and gradient checkpointing (`jax.checkpoint`) to navigate stiff, long-time ionic chains.
3. **Automated Parameter Optimization**: Blindly recovers hidden physiological settings using a multi-stage curriculum.

---

## Why the Project is Useful

Traditional cardiac solvers lack efficient, native integration with machine learning or modern optimization tooling. By implementing the physics graph entirely within JAX, this framework allows researchers to:
* Perform high-throughput forward simulations on modern GPU hardware.
* Execute exact parameter estimation without relying on manual trial-and-error tuning curves.
* Maintain complete data tracking under strict blind optimization protocols.

---

## How Users Can Get Started

All code blocks run out of a single integrated Python setup. Use the instructions below to configure and run the simulation suite.

### 1. Conda Environment Setup
Navigate to your project root folder and build the environment using the provided commands:

```bash
# Move to repository root
cd path_to_dir\Github_repo

# Create and activate environment
conda create --name jax_env python=3.10 -y
conda activate jax_env

# Install package dependencies
pip install -r requirements.txt
```

### 2. Relative Directory Paths and Execution
Each component runs independently from within its own local subdirectory. Always ensure your environment is active before running scripts:

* **To run validation against bench trace (0D) and CARP reference (2D)[3]:**
  ```bash
  cd benchmark/0D_validation/
  python 0D_differentiable_validation.py
  ```
  ```bash
  cd benchmark/2D_validation/
  python benchmark2D.py
  ```
* **To run the left atrial 2D tissue sample and compare CPU/GPU performance (forward simulations):**
  ```bash
  cd forward_model/
  python run_forward.py
  ```
   ```bash
  cd forward_model/
  python run_cpu_gpu_comparison.py
  ```
* **To run the parameter learning framework:**
  ```bash
  cd parameter_learning/
  python Run_GT_generation.py
  python Parameter_learning.py
  ```

---

## Technical Specifications and Performance 

### MRI derived Patient Specific Left atrial (LA) Forward Simulation Performance (NVIDIA Tesla T4)
* **CS Pacing Forward:** 46.2s GPU vs. 2608.2s CPU (**56x Speedup**)
* **LAA Pacing Forward:** 46.8s GPU vs. 2738.6s CPU (**58x Speedup**)
* **Full Jacobian Matrix ($dAT/dG_{IL}$):** 90.4s GPU vs. 5096.2s CPU (**56x Speedup**)

### 3D Tissue-Sample Cube Forward Performance (NVIDIA Tesla P100)
To ensure a comparable test with similar spatial and temporal resolution as compared to the patient LA mesh (dx = 0.351mm median edge length, DT = 0.1ms), the cube uses dx = 0.2mm and DT = 0.1ms with N_T = 11,100 timesteps
* **Forward Simulation:** 57.7s CPU vs. 1.34s GPU (**43× Speedup**)
* **Activation Map:** Identical CPU and GPU activation patterns
* **Maximum |AT Difference|:** 0.000 ms

<table>
  <tr>
    <th>Metric</th>
    <th>CPU</th>
    <th>GPU</th>
    <th>GPU Speed-up</th>
  </tr>
  <tr>
    <td><b>Forward simulation</b></td>
    <td>57.732 s</td>
    <td><b>1.342 s</b></td>
    <td><b>43.0×</b></td>
  </tr>
  <tr>
    <td>Activated nodes</td>
    <td>18,432</td>
    <td>18,432</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Activation-time range</td>
    <td>13.10–54.40 ms</td>
    <td>13.10–54.40 ms</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Activation map agreement</td>
    <td>—</td>
    <td><b>Identical</b></td>
    <td><b>0.000 ms</b> max difference</td>
  </tr>
</table>

The GPU reduced the forward simulation time by **43×** while producing an activation map numerically identical to the CPU implementation.
### Parameter Inversion Results (NVIDIA Tesla P100)
The framework optimizes parameters under strict blind criteria using an S1-S2 pacing protocol. The tracking engine uses a 7-level Nelder-Mead shape-fitting curriculum followed by L-BFGS-B gradient refinement.
* **Mean Absolute Percentage Error (MAPE):** 0.676% across all tested regimes (`set1`, `set2`, `set3`).
* **Median Absolute Percentage Error:** 0.032%
* **Exact Recovery:** Recovers `tau_in` and `G_IL` exactly to 3 decimal places across all trials.

| Case | Phase 1 Loss | Phase 2 Loss | Evals (P2) | Wall Time |
| :--- | :----------- | :----------- | :--------- | :-------- |
| **set1** | 0.000094     | 1.15e-09     | 18         | 5.7 min   |
| **set2** | 0.000025     | 1.09e-09     | 50         | 13.7 min  |
| **set3** | 0.000045     | 2.95e-07     | 11         | 3.3 min   |

---

## Further Reading

For individual component deep-dives, mathematical validations, structural nuances, and complete data formats, refer to the relevant text files located directly within each subfolder.

## Further Reading

For individual component deep-dives, mathematical validations, structural nuances, and complete data formats, refer to the relevant text files located directly within each subfolder.

### References

[1] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S. & Zhang, Q. (2018). *JAX: composable transformations of Python+NumPy programs*. Version 0.3.13. https://github.com/google/jax

[2] Corrado, C. & Niederer, S. A. (2016). *A two-variable model robust to pacemaker behaviour for the dynamics of the cardiac action potential*. **Mathematical Biosciences, 281**, 46–54.

[3] Plank, G., Loewe, A., Neic, A., Augustin, C., Huang, Y.-L., Gsell, M. A. F., Karabelas, E., Nothstein, M., Prassl, A. J., Sánchez, J. & others (2021). *The openCARP simulation environment for cardiac electrophysiology*. **Computer Methods and Programs in Biomedicine, 208**, 106223.
