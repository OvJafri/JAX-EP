# -*- coding: utf-8 -*-
"""
run_cpu_gpu_comparison.py
=============================
SINGLE, self-contained script: run both the CPU and GPU forward-pass
benchmarks and print a comparison table.

"""
import os
import sys
import subprocess
import time

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
DATASET_PATH = '/path_to_folder/cube_validation_3d'
WORK_DIR = "/path_to_folder/working" if os.path.isdir("/path_to_folder/working") else "."
# ══════════════════════════════════════════════════════════════════════════

OUT_DIR = os.path.join(WORK_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
WORKER_PATH = os.path.join(WORK_DIR, "_cube_worker_tmp.py")

# ── The worker script, embedded as a string. Identical physics/logic
#    to the already-validated run_gpu_cpu_benchmark.py -- only
#    packaged differently (as a string written out at runtime,
#    instead of a separately uploaded file) so this whole benchmark
#    is a single, self-contained script. ─────────────────────────────────
WORKER_SOURCE = '''# -*- coding: utf-8 -*-
import os, sys, time, subprocess
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DATASET_PATH = "{folder_path}"
if os.path.isdir(DATASET_PATH) and DATASET_PATH not in sys.path:
    sys.path.insert(0, DATASET_PATH)

import cube_validation_3d as cube3d
from cube_validation_3d.solver import make_forward_solver_atmap

OUT_DIR = "{out_dir}"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = ""

devices = jax.devices()
backend = jax.default_backend()
device_tag = "gpu" if backend == "gpu" else "cpu"

print("="*60)
print(f"Worker running on: {{device_tag.upper()}}")
if gpu_info:
    print(f"  GPU: {{gpu_info}}")
print(f"  Devices: {{devices}}  Backend: {{backend}}")
print("="*60)

LX_MM, LY_MM, THICKNESS_MM, DX_MM = 19.0, 19.0, 1.5, 0.2
PACE_EDGE = "LEFT"
V_GATE, A_CRIT, BETA, CM = 0.13, 0.13, 140.0, 1.0
G_IL_NOMINAL = 0.350
STIM_AMP, STIM_DUR = 200.0, 2.0
N_ION, DT = 4, 0.1
S1_START, S2_BCL = 10.0, 500.0
S2_ON = S1_START + S2_BCL
TOTAL_MS = S2_ON + 600.0
NT = int(TOTAL_MS / DT)

t0 = time.time()
Verts, Elems, top_grid, bot_grid = cube3d.build_thin_plate_mesh(
    LX_MM, LY_MM, THICKNESS_MM, DX_MM)
mask_pace, paced_ids = cube3d.build_edge_pacing_mask(
    Verts, top_grid, bot_grid, edge=PACE_EDGE)
m_inv, eu, ev, ecot, ed_x = cube3d.build_fem_operators(Verts, Elems, CM=CM)
w = cube3d.anisotropic_weights(ecot, ed_x, G_IL=G_IL_NOMINAL, BETA=BETA, CM=CM)
print(f"  {{len(Verts):,}} nodes  {{len(eu):,}} edges  ({{time.time()-t0:.1f}}s)")

t_ms = np.arange(NT) * DT
sv = (t_ms >= S1_START) & (t_ms < S1_START + STIM_DUR)

run_atmap = make_forward_solver_atmap(
    Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, w=w, mask_pace=mask_pace,
    V_GATE=V_GATE, A_CRIT=A_CRIT, DT=DT, N_ION=N_ION, STIM_AMP=STIM_AMP,
    CM=CM, sv_schedule=sv)

p5_nominal = jnp.array(
    [0.300, 5.000, 120.0, 150.0, G_IL_NOMINAL], dtype=jnp.float64)

t0 = time.time()
_ = run_atmap(p5_nominal)
jax.block_until_ready(_)
t_warmup = time.time() - t0
print(f"  Warm-up: {{t_warmup:.2f}}s")

t0 = time.time()
at_hard, at_soft, activated = run_atmap(p5_nominal)
at_hard.block_until_ready()
t_forward = time.time() - t0
print(f"  Forward pass (warm): {{t_forward:.3f}}s")

at_hard_np = np.array(at_hard)
at_soft_np = np.array(at_soft)
act_np = np.array(activated)
print(f"  Activated: {{act_np.sum():,}}/{{len(Verts):,}} nodes")

out_path = os.path.join(OUT_DIR, f"cube_benchmark_{{device_tag}}.npz")
np.savez(out_path,
         at_hard=at_hard_np, at_soft=at_soft_np, activated=act_np,
         t_forward=t_forward, t_warmup=t_warmup, device_tag=device_tag)
print(f"Saved: {{out_path}}")
print("WORKER_DONE")
'''

WORKER_SOURCE = WORKER_SOURCE.format(folder_path=DATASET_PATH, out_dir=OUT_DIR)

with open(WORKER_PATH, "w") as f:
    f.write(WORKER_SOURCE)
print(f"Wrote worker to: {WORKER_PATH}")


def run_worker(force_cpu):
    label = "CPU" if force_cpu else "GPU"
    print(f"\n{'='*60}")
    print(f"Running {label} pass (subprocess) ...")
    print(f"{'='*60}")

    env = os.environ.copy()
    if force_cpu:
        env["JAX_PLATFORMS"] = "cpu"
    elif "JAX_PLATFORMS" in env:
        del env["JAX_PLATFORMS"]

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, WORKER_PATH], env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    if result.returncode != 0 or "WORKER_DONE" not in result.stdout:
        raise RuntimeError(
            f"{label} subprocess failed (exit code {result.returncode}) "
            f"-- see output above.")

    print(f"  [{label} subprocess wall-clock: {elapsed:.1f}s total]")


run_worker(force_cpu=True)
run_worker(force_cpu=False)

# ── Load both results and build the table ─────────────────────────────────
cpu_path = os.path.join(OUT_DIR, "cube_benchmark_cpu.npz")
gpu_path = os.path.join(OUT_DIR, "cube_benchmark_gpu.npz")

if not os.path.exists(cpu_path):
    raise RuntimeError(f"CPU result not found at {cpu_path}")
if not os.path.exists(gpu_path):
    raise RuntimeError(
        f"GPU result not found at {gpu_path} -- is a GPU actually "
        f"available/enabled in this session's settings?")

cpu_data = np.load(cpu_path)
gpu_data = np.load(gpu_path)

t_cpu = float(cpu_data["t_forward"])
t_gpu = float(gpu_data["t_forward"])
speedup = t_cpu / t_gpu

at_hard_cpu = cpu_data["at_hard"]
at_hard_gpu = gpu_data["at_hard"]
activated_cpu = cpu_data["activated"]
activated_gpu = gpu_data["activated"]

both_activated = activated_cpu & activated_gpu
if both_activated.sum() > 0:
    at_diff = np.abs(at_hard_cpu[both_activated] - at_hard_gpu[both_activated])
    max_at_diff = float(at_diff.max())
    mean_at_diff = float(at_diff.mean())
else:
    max_at_diff = mean_at_diff = float("nan")

activation_match = bool(np.array_equal(activated_cpu, activated_gpu))

print(f"\n{'='*60}")
print("COMPARISON TABLE")
print(f"{'='*60}")
print(f"\n  {'Metric':<32}{'CPU':>14}{'GPU':>14}")
print(f"  {'-'*60}")
print(f"  {'Forward pass (warm, s)':<32}{t_cpu:>14.3f}{t_gpu:>14.3f}")
print(f"  {'Nodes activated':<32}{int(activated_cpu.sum()):>14,}"
      f"{int(activated_gpu.sum()):>14,}")
print(f"  {'AT range min (ms)':<32}{at_hard_cpu[activated_cpu].min():>14.2f}"
      f"{at_hard_gpu[activated_gpu].min():>14.2f}")
print(f"  {'AT range max (ms)':<32}{at_hard_cpu[activated_cpu].max():>14.2f}"
      f"{at_hard_gpu[activated_gpu].max():>14.2f}")

print(f"\n  {'Speedup (CPU time / GPU time)':<32}{speedup:>28.1f}x")

print(f"\n  {'Activation pattern identical':<32}{str(activation_match):>28}")
print(f"  {'Mean |AT difference| (ms)':<32}{mean_at_diff:>28.4f}")
print(f"  {'Max |AT difference| (ms)':<32}{max_at_diff:>28.4f}")

print(f"\n{'='*60}")
consistent = 0.5 * 40 < speedup < 2.0 * 40
identical_enough = activation_match and max_at_diff < 0.5
print(f"  Speedup consistent with manuscript's ~40x claim: "
      f"{'YES' if consistent else 'NO'}")
print(f"  Activation maps effectively identical (< 0.5ms max diff): "
      f"{'YES' if identical_enough else 'NO'}")
print(f"{'='*60}")

print("\nDONE")