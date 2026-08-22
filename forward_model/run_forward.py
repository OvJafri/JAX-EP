# -*- coding: utf-8 -*-
"""
run_forward.py
=======================
STEP 1 of 2: validates the core forward physics (wave propagation,
activation-time map, conduction velocity) on the full-scale 3D
thin-plate cube, WITHOUT the HD-grid/lead-field/EGM calculation.

This deliberately isolates the two concerns that were previously
tangled together (and where two genuine bugs were already found in
the lead-field/electrode code): core wave-propagation physics vs.
electrode/lead-field computation. Run this FIRST; only once this
looks physiologically sensible should you move to
run_cube_benchmark.py (which adds the HD grid, OEGMs, and the
FD/AD/jacfwd differentiability comparison).

Uses exactly the same mesh, FEM, and forward-solver settings as
run_cube_benchmark.py (dx=0.2mm, DT=0.1ms, N_ION=4, Rush-Larsen) --
only the output (activation-time map instead of lead-field/EGM time
series) differs.

The output file `cube_forward_only_atmap.npz` can be post-processed
using `visualization_forward.py` to generate the corresponding
2D activation-time (LAT) figure for visual inspection of the
propagation pattern.
"""
import os
import time
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import cube_validation_3d as cube3d
from cube_validation_3d.solver import make_forward_solver_atmap

# ── Paths ─────────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", "./outputs")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu_info}")
except Exception:
    pass
print("=" * 60)
print("STEP 1/2 — Forward-only physics validation (no HD grid/EGM)")
print(f"  JAX:     {jax.__version__}")
print(f"  Devices: {jax.devices()}")
print("=" * 60)

# ── Geometry / physics (matches run_cube_benchmark.py exactly) ───────────
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

print(f"\n  Geometry: {LX_MM}x{LY_MM}x{THICKNESS_MM}mm, dx={DX_MM}mm")
print(f"  Protocol: DT={DT}ms  NT={NT}  TOTAL={TOTAL_MS}ms")
print(f"  Pacing:   entire {PACE_EDGE} edge")

# ── Build mesh / FEM (identical to run_cube_benchmark.py) ────────────────
print("\n[1] Building thin-plate mesh ...")
t0 = time.time()
Verts, Elems, top_grid, bot_grid = cube3d.build_thin_plate_mesh(
    LX_MM, LY_MM, THICKNESS_MM, DX_MM)
print(f"  {len(Verts):,} nodes  {len(Elems):,} elements  ({time.time()-t0:.1f}s)")

mask_pace, paced_ids = cube3d.build_edge_pacing_mask(
    Verts, top_grid, bot_grid, edge=PACE_EDGE)
print(f"  Pacing mask: {int(mask_pace.sum())} nodes ({PACE_EDGE} edge, top+bottom)")

print("\n[2] Building FEM operators ...")
t0 = time.time()
m_inv, eu, ev, ecot, ed_x = cube3d.build_fem_operators(Verts, Elems, CM=CM)
w = cube3d.anisotropic_weights(ecot, ed_x, G_IL=G_IL_NOMINAL, BETA=BETA, CM=CM)
print(f"  {len(eu):,} edges  ({time.time()-t0:.1f}s)")

# ── S1-only stimulus schedule (single stimulus is enough to validate
#    propagation; S2 isn't needed for this physics-only check) ───────────
t_ms = np.arange(NT) * DT
sv = (t_ms >= S1_START) & (t_ms < S1_START + STIM_DUR)
print(f"\n  Stimulus: {sv.sum()} timesteps = {sv.sum()*DT}ms "
      f"(should equal STIM_DUR={STIM_DUR}ms)")
assert abs(sv.sum() * DT - STIM_DUR) < 1e-9, "Stimulus duration mismatch!"

# ── Build and run the activation-time-map solver ──────────────────────────
run_atmap = make_forward_solver_atmap(
    Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, w=w, mask_pace=mask_pace,
    V_GATE=V_GATE, A_CRIT=A_CRIT, DT=DT, N_ION=N_ION, STIM_AMP=STIM_AMP,
    CM=CM, sv_schedule=sv)

p5_nominal = jnp.array(
    [0.300, 5.000, 120.0, 150.0, G_IL_NOMINAL], dtype=jnp.float64)

print("\n[3] Running forward pass (warm-up + timed) ...")
_ = run_atmap(p5_nominal)  # warm up JIT
t0 = time.time()
at_hard, at_soft, activated = run_atmap(p5_nominal)
at_hard.block_until_ready()
t_forward = time.time() - t0
print(f"  Forward (warm): {t_forward:.2f}s")

at_np = np.array(at_hard)
at_soft_np = np.array(at_soft)
act_np = np.array(activated)

# ── Physics sanity checks ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHYSICS SANITY CHECKS")
print(f"{'='*60}")

print(f"\n  Activated: {act_np.sum():,}/{len(Verts):,} nodes")
if act_np.sum() == 0:
    print("  FAIL: NO nodes activated at all -- stimulus did not capture.")
else:
    print(f"  AT range: [{at_np[act_np].min():.2f}, {at_np[act_np].max():.2f}]ms")

    # Check monotonic-ish increase in AT with distance from the paced edge
    x_coords = Verts[:, 0]
    corr = np.corrcoef(x_coords[act_np], at_np[act_np])[0, 1]
    print(f"\n  Correlation (x-distance from paced edge) vs (activation time): "
          f"{corr:.4f}")
    print(f"  {'PASS' if corr > 0.8 else 'FAIL'}: wave propagates "
          f"{'sensibly outward' if corr > 0.8 else 'NOT sensibly -- check mesh/pacing/params'} "
          f"from the paced edge")

    # Rough conduction velocity estimate along the plate's long axis,
    # away from the immediate pacing-site region
    y0 = Verts[:, 1].min()
    strip = (np.abs(Verts[:, 1] - (y0 + LY_MM / 2)) < 1.0) & \
            (Verts[:, 0] > 2.0) & act_np
    if strip.sum() > 2:
        xs = x_coords[strip]
        ats = at_np[strip]
        order = np.argsort(xs)
        xs, ats = xs[order], ats[order]
        dx_mm = xs[-1] - xs[0]
        dt_ms = ats[-1] - ats[0]
        if dt_ms > 0.5:
            cv_m_s = (dx_mm * 1e-3) / (dt_ms * 1e-3)
            print(f"\n  Rough conduction velocity estimate: {cv_m_s:.3f} m/s")
            print(f"  (compare to the manuscript's own reported "
                  f"CV~0.384-0.471 m/s range for this ionic model)")

# ── Save ──────────────────────────────────────────────────────────────────
np.savez(os.path.join(OUT_DIR, "cube_forward_only_atmap.npz"),
         at_hard=at_np, at_soft=at_soft_np, activated=act_np,
         Verts=Verts, Elems=Elems, DT=DT, NT=NT)
print(f"\nSaved: cube_forward_only_atmap.npz")
print("\nDONE — if the checks above look sensible, proceed to "
      "run_cube_benchmark.py for the HD-grid/EGM/differentiability step.")