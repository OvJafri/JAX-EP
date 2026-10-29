# -*- coding: utf-8 -*-
import os, sys, time, subprocess
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DATASET_PATH = "/data/home/exx915/Acutus_upload/Forward_model"
if os.path.isdir(DATASET_PATH) and DATASET_PATH not in sys.path:
    sys.path.insert(0, DATASET_PATH)

import cube_validation_3d as cube3d
from cube_validation_3d import build_atmap_solver

IONIC_MODEL = "CRN"
OUT_DIR = "./outputs"
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
print(f"Worker running on: {device_tag.upper()}  (ionic model: {IONIC_MODEL})")
if gpu_info:
    print(f"  GPU: {gpu_info}")
print(f"  Devices: {devices}  Backend: {backend}")
print("="*60)

LX_MM, LY_MM, THICKNESS_MM, DX_MM = 19.0, 19.0, 1.5, 0.2
PACE_EDGE = "LEFT"
BETA, CM = 140.0, 1.0
G_IL_NOMINAL = 0.350
STIM_DUR = 2.0
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
print(f"  {len(Verts):,} nodes  {len(eu):,} edges  ({time.time()-t0:.1f}s)")

t_ms = np.arange(NT) * DT
sv = (t_ms >= S1_START) & (t_ms < S1_START + STIM_DUR)

run_atmap, p_nominal, settings = build_atmap_solver(
    IONIC_MODEL, Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, ecot=ecot, ed_x=ed_x,
    mask_pace=mask_pace, BETA=BETA, CM=CM, DT=DT, N_ION=N_ION, sv_schedule=sv,
    G_IL=G_IL_NOMINAL)

t0 = time.time()
_ = run_atmap(p_nominal)
jax.block_until_ready(_)
t_warmup = time.time() - t0
print(f"  Warm-up: {t_warmup:.2f}s")

t0 = time.time()
at_hard, at_soft, activated = run_atmap(p_nominal)
at_hard.block_until_ready()
t_forward = time.time() - t0
print(f"  Forward pass (warm): {t_forward:.3f}s")

at_hard_np = np.array(at_hard)
at_soft_np = np.array(at_soft)
act_np = np.array(activated)
print(f"  Activated: {act_np.sum():,}/{len(Verts):,} nodes")

out_path = os.path.join(OUT_DIR, f"cube_benchmark_{IONIC_MODEL}_{device_tag}.npz")
np.savez(out_path,
         at_hard=at_hard_np, at_soft=at_soft_np, activated=act_np,
         t_forward=t_forward, t_warmup=t_warmup, device_tag=device_tag)
print(f"Saved: {out_path}")
print("WORKER_DONE")
