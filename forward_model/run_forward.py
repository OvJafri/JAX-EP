# -*- coding: utf-8 -*-
"""
run_forward.py
==============
Forward-only physics check on the 3D thin-plate cube: wave propagation,
activation-time (AT) map and conduction velocity, without the HD-grid /
EGM calculation.

IONIC MODEL TOGGLE
------------------
Set IONIC_MODEL below (or the environment variable IONIC_MODEL) to
    "mMS"  modified Mitchell-Schaeffer   p = [tau_in, tau_out, tau_open, tau_close, G_IL]
    "CRN"  Courtemanche-Ramirez-Nattel   p = [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL]
Per-model settings (stimulus amplitude, AT threshold, nominal parameters)
are in cube_validation_3d/ionic_models.py.

Mesh/FEM/time stepping are the same for both models (dx = 0.2 mm,
DT = 0.1 ms, N_ION = 4, Crank-Nicolson diffusion, BETA = 140).

Output: outputs/cube_forward_only_atmap_<IONIC_MODEL>.npz, post-processed by
visualize_forward.py (2D LAT/CV figure) and visualize_3d_plotly.py
(interactive 3D view).
"""
import os
import time
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import cube_validation_3d as cube3d
from cube_validation_3d import build_atmap_solver

# ══════════════════════════════════════════════════════════════════════════
IONIC_MODEL = os.environ.get("IONIC_MODEL", "mMS")     # "mMS" or "CRN"
# ══════════════════════════════════════════════════════════════════════════

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
print(f"Forward-only physics check (no HD grid/EGM) — ionic model: {IONIC_MODEL}")
print(f"  JAX:     {jax.__version__}")
print(f"  Devices: {jax.devices()}")
print("=" * 60)

# ── Geometry / physics (shared by both ionic models) ─────────────────────
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

print(f"\n  Geometry: {LX_MM}x{LY_MM}x{THICKNESS_MM}mm, dx={DX_MM}mm")
print(f"  Protocol: DT={DT}ms  NT={NT}  TOTAL={TOTAL_MS}ms")
print(f"  Pacing:   entire {PACE_EDGE} edge")

# ── Build mesh / FEM ─────────────────────────────────────────────────────
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
print(f"  {len(eu):,} edges  ({time.time()-t0:.1f}s)")

# ── S1-only stimulus schedule (single stimulus is enough to validate
#    propagation; S2 isn't needed for this physics-only check) ───────────
t_ms = np.arange(NT) * DT
sv = (t_ms >= S1_START) & (t_ms < S1_START + STIM_DUR)
print(f"\n  Stimulus: {sv.sum()} timesteps = {sv.sum()*DT}ms "
      f"(should equal STIM_DUR={STIM_DUR}ms)")
assert abs(sv.sum() * DT - STIM_DUR) < 1e-9, "Stimulus duration mismatch!"

# ── Build and run the activation-time-map solver ──────────────────────────
run_atmap, p_nominal, settings = build_atmap_solver(
    IONIC_MODEL, Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, ecot=ecot, ed_x=ed_x,
    mask_pace=mask_pace, BETA=BETA, CM=CM, DT=DT, N_ION=N_ION, sv_schedule=sv,
    G_IL=G_IL_NOMINAL)
print(f"  Ionic model: {IONIC_MODEL}  stimulus {settings['STIM_AMP']}  "
      f"AT threshold V_norm > {settings['V_GATE']}")
print("  Parameters: " + ", ".join(
    f"{n}={float(v):g}" for n, v in zip(settings["PNAMES"], p_nominal)))

print("\n[3] Running forward pass (warm-up + timed) ...")
_ = run_atmap(p_nominal)  # warm up JIT
t0 = time.time()
at_hard, at_soft, activated = run_atmap(p_nominal)
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
            if IONIC_MODEL == "mMS":
                print(f"  (compare to the manuscript's own reported "
                      f"CV~0.384-0.471 m/s range for this ionic model)")

# ── Save ──────────────────────────────────────────────────────────────────
out_name = f"cube_forward_only_atmap_{IONIC_MODEL}.npz"
np.savez(os.path.join(OUT_DIR, out_name),
         at_hard=at_np, at_soft=at_soft_np, activated=act_np,
         Verts=Verts, Elems=Elems, DT=DT, NT=NT,
         ionic_model=IONIC_MODEL, params=np.array(p_nominal),
         LX_MM=LX_MM, LY_MM=LY_MM, DX_MM=DX_MM)
print(f"\nSaved: {out_name}")
print("\nDONE — visualise with visualize_forward.py and visualize_3d_plotly.py "
      f"(same IONIC_MODEL={IONIC_MODEL}).")