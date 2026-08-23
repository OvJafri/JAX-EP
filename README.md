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


JAX-EP brings scalable GPU-accelerated cardiac monodomain modelling to gradient-based parameter learning, exposing gradients of EP model outputs with respect to model parameters. It achieves:
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

* **To run JAX-EP solver validation against carpentry bench (0D trace) and CARP 2D reference simulation [3]:**
  ```bash
  cd benchmark/0D_validation/
  python 0D_differentiable_validation.py
  ```
  ```bash
  cd benchmark/2D_validation/
  python benchmark2D.py
  ```
* **To run LA tissue sample (3D cube) using JAX-EP solver and compare CPU/GPU performance (forward simulations):**
  ```bash
  cd forward_model/
  python run_forward.py
  python run_cpu_gpu_comparison.py
  ```
* **To evaluate gradients using JAX-EP end-to-end differentiable simulations, i.e. $\partial\mathcal{L}/\partial\theta$:**
  ```bash
  cd differentiability/3D_cube
  python run_differentiability_cpu_gpu.py

  cd differentiability/LA_patch
  python patch_differentiability_showcase.py
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

## Solver Differentiability

A central feature of JAX-EP is its end-to-end differentiability, allowing gradients to be propagated through the complete cardiac electrophysiology simulation pipeline. This enables derivatives of simulated electrophysiological observables with respect to cellular and tissue parameters, providing the basis for gradient-based parameter inference and personalised electrophysiological model calibration.

<img src="differentiability/figure_cube_gradient_verification.png" alt="JAX-EP Forward-mode Automatic Differentiation Validation" width="650">

Above differentiability results can be reproduced using `run_differentiability_cpu_gpu.py`, which uses a 3D cube representative of the full LA simulation, with matching spatial and temporal resolutions. The example uses a 3D tissue cube comprising **18,432 nodes and 36,860 elements**, with a spatial resolution of `dx = 0.2 mm`, a time step of `DT = 0.1 ms`, and `N_T = 11,100` time steps (total simulation time: 1110 ms). The script compares finite-difference (FD) gradients with reverse-mode (`jax.grad`) and forward-mode (`jax.jacfwd`) automatic differentiation, and includes the corresponding CPU/GPU benchmark.

### Forward-Mode Autodiff

For the tested ionic parameters (`tau_in`, `tau_out`, `tau_open`, `tau_close`) and longitudinal intracellular conductivity (`G_IL`), `jax.jacfwd` reproduced the finite-difference gradients with excellent agreement:

| Parameter | FD Gradient | `jax.jacfwd` | Ratio |
| :--- | ---: | ---: | ---: |
| `tau_in` | -3.0118e-01 | -3.0116e-01 | 0.9999 |
| `tau_out` | 1.8851e-02 | 1.8851e-02 | 1.0000 |
| `tau_open` | -6.1344e-10 | -6.2751e-10 | 1.0229 |
| `tau_close` | 1.0544e-03 | 1.0544e-03 | 1.0000 |
| `G_IL` | -3.1939e-04 | -3.2335e-04 | 1.0124 |

The forward-mode derivatives were stable across all tested parameters, with the largest deviation from the finite-difference reference of approximately **2.3%**. These gradient comparisons can be reproduced and visualised using `plot_gradients.py`.

### Reverse-Mode Autodiff

In contrast, direct reverse-mode differentiation using `jax.grad` produced severe gradient amplification for the stiff ionic dynamics. The resulting derivatives differed from the finite-difference reference by several orders of magnitude, making the gradients unsuitable for stable parameter optimization in this configuration.

This behaviour was observed consistently across both CPU and GPU executions, indicating that the issue originates from the differentiation of the stiff ionic time integration rather than the computational device.

### Differentiable Electrogram Generation

The differentiability extends beyond the transmembrane voltage state to derived electrophysiological observables. A bipolar EGM configuration using two spatially separated electrode regions (0.3 mm inter electrode gap) was included in the benchmark. Forward-mode differentiation again reproduced the finite-difference gradients with close agreement, including for `G_IL`.

| Parameter | FD Gradient | `jax.jacfwd` | Ratio |
| :--- | ---: | ---: | ---: |
| `tau_in` | 9.7765e-03 | 9.7775e-03 | 1.0001 |
| `tau_out` | -8.7719e-05 | -8.7848e-05 | 1.0015 |
| `tau_open` | -2.1321e-09 | -2.1333e-09 | 1.0006 |
| `tau_close` | -1.5596e-06 | -1.5651e-06 | 1.0035 |
| `G_IL` | -1.2619e-02 | -1.2620e-02 | 1.0001 |

The bipolar EGM benchmark therefore provides an independent demonstration that gradients can be propagated through the simulation and EGM generation pipeline.

### CPU/GPU Differentiability Performance

The differentiability benchmark also demonstrates the computational advantage of JAX-EP on GPU hardware. For the complete FD, `jax.grad`, and `jax.jacfwd` benchmark:

| Benchmark | CPU | GPU | GPU Speed-up |
| :--- | ---: | ---: | ---: |
| Raw voltage (`V²` loss) | 1722.8 s | 45.6 s | **37.8×** |
| Bipolar EGM | 1866.9 s | 45.9 s | **40.7×** |

The GPU implementation therefore provides approximately **38–41× acceleration** for the differentiability benchmark while retaining agreement with the reference finite-difference gradients.

### Patient-Specific LA Patch Differentiability

The differentiability workflow was also tested on a representative patch extracted from the **patient-specific MRI-derived LA anatomy** used for the HD-grid omnipolar EGM simulations. The extracted patch contains **2,894 nodes and 5,572 elements** and retains the same spatial and temporal discretisation used in the full LA simulation. The patch also includes the HD-grid electrode geometry and fibre information required to generate the simulated omnipolar EGMs.

The complete test can be reproduced using `patch_differentiability_showcase.py` with the provided `patch_geometry.npz`. The script generates the ground-truth omnipolar EGMs and compares finite-difference (FD), reverse-mode (`jax.grad`), and forward-mode (`jax.jacfwd`) derivatives for the five model parameters: `tau_in`, `tau_out`, `tau_open`, `tau_close`, and `G_IL`.

For the representative `set3` case, `jax.jacfwd` remained stable for all five parameters and closely matched the FD reference:

| Parameter | FD Gradient | `jax.jacfwd` | Ratio |
| :--- | ---: | ---: | ---: |
| `tau_in` | -2.6556e+03 | -2.6730e+03 | 1.0066 |
| `tau_out` | 1.8406e+01 | 1.9386e+01 | 1.0533 |
| `tau_open` | 2.6224e+00 | 2.6142e+00 | 0.9969 |
| `tau_close` | 2.7259e+00 | 2.7538e+00 | 1.0103 |
| `G_IL` | -2.7231e+04 | -2.7209e+04 | 0.9992 |

Forward-mode derivatives agreed closely with the FD reference across all five parameters — mean absolute deviation ~1.5%, maximum ~5.3%, both within the benchmark's stability criterion — whereas reverse-mode jax.grad again diverged from the FD reference by many orders of magnitude due to the stiff ionic dynamics

This example extends the differentiability test beyond the simplified 3D cube to a geometry **extracted directly from the patient-specific LA anatomy**, while retaining the simulated HD-grid omnipolar EGM generation pipeline. The patch test and outputs can be reproduced with `patch_differentiability_showcase.py`; the extracted geometry is provided as `patch_geometry.npz`.
<img src="differentiability/newplot_LA.png" alt="JAX-EP Patient specific LA patch differentiability showcase" width="650">

Overall, these tests establish that JAX-EP provides a **fully differentiable cardiac EP simulation pipeline**, with forward-mode automatic differentiation providing stable gradients through the stiff ionic dynamics and derived EGM calculations. This differentiability forms the computational basis for the subsequent **gradient-based parameter-learning and personalised EP inference** framework.



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

* R. Newbury *et al.*, "**A Review of Differentiable Simulators**," in *IEEE Access*, vol. 12, pp. 98114-98132, 2024, doi: [10.1109/ACCESS.2024.3425448](https://doi.org).


### References

[1] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S. & Zhang, Q. (2018). *JAX: composable transformations of Python+NumPy programs*. Version 0.3.13. https://github.com/google/jax

[2] Corrado, C. & Niederer, S. A. (2016). *A two-variable model robust to pacemaker behaviour for the dynamics of the cardiac action potential*. **Mathematical Biosciences, 281**, 46–54.

[3] Plank, G., Loewe, A., Neic, A., Augustin, C., Huang, Y.-L., Gsell, M. A. F., Karabelas, E., Nothstein, M., Prassl, A. J., Sánchez, J. & others (2021). *The openCARP simulation environment for cardiac electrophysiology*. **Computer Methods and Programs in Biomedicine, 208**, 106223.
