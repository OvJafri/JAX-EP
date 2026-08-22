# -*- coding: utf-8 -*-
"""
visualize_forward.py
=========================
Visualizes the output of run_forward_only.py: a top-face activation
(LAT) map with isochrones and a propagation-direction indicator, a
measured conduction-velocity profile checked directly against the
3D LA forward simulation (patient MRI-derived mesh)'s known CV
range, and a hard-vs-soft activation-time agreement check -- for the
3D thin-plate cube's core-physics validation step.

Panel B is the one that carries this step's validation message
directly and visually: a conduction-velocity profile (activation
time vs distance from the paced edge), with a fitted CV line plotted
against the 3D LA forward simulation's own reported CV range, rather
than leaving that comparison as printed text only.

Only the TOP face is plotted -- not the full 3D shell (top, bottom,
and side walls) -- since projecting the full closed shell onto a
flat 2D view would overlap the top and bottom faces at similar
(x, y) coordinates. The top-face triangulation is reconstructed here
directly from the same regular-grid node ordering used in mesh.py's
build_thin_plate_mesh (verified to exactly match the actual mesh's
own top-face triangles, not just independently plausible).

Usage
-----
    python visualize_forward.py

Reads cube_forward_only_atmap.npz from OUT_DIR (see below) and writes
figure_cube_forward.png alongside it.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ── Paths ────────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", "./outputs")
IN_PATH = os.path.join(OUT_DIR, "cube_forward_only_atmap.npz")

# Must match the geometry actually used in run_forward_only.py
LX_MM, LY_MM, THICKNESS_MM, DX_MM = 19.0, 19.0, 1.5, 0.2

# 3D LA forward simulation (patient MRI-derived mesh)'s reported
# conduction-velocity range (Full-LA, CS/LAA pacing) -- shown directly
# on the CV-profile panel for a genuine, visual comparison
CLINICAL_CV_MIN_MS = 0.384  # m/s
CLINICAL_CV_MAX_MS = 0.471  # m/s

print(f"Loading {IN_PATH} ...")
data = np.load(IN_PATH)
Verts = data["Verts"]
at_hard = data["at_hard"]
at_soft = data["at_soft"]
activated = data["activated"]
DT = float(data["DT"])
print(f"  {len(Verts):,} nodes  DT={DT}ms")

# ── Reconstruct the top-face triangulation only ────────────────────────────
nx = int(round(LX_MM / DX_MM)) + 1
ny = int(round(LY_MM / DX_MM)) + 1
n_per_face = nx * ny
top_grid = np.arange(n_per_face).reshape(ny, nx)  # matches mesh.py exactly
top_tris = []
for iy in range(ny - 1):
    for ix in range(nx - 1):
        bl, br = top_grid[iy, ix], top_grid[iy, ix + 1]
        tl, tr = top_grid[iy + 1, ix], top_grid[iy + 1, ix + 1]
        top_tris.append([bl, br, tr]); top_tris.append([bl, tr, tl])
top_tris = np.array(top_tris, dtype=np.int64)
print(f"  Top-face triangulation: {len(top_tris):,} triangles")

top_node_ids = top_grid.ravel()
top_x = Verts[top_node_ids, 0]
top_y = Verts[top_node_ids, 1]
triang = Triangulation(top_x, top_y, top_tris)

at_hard_top = np.where(activated[:n_per_face], at_hard[:n_per_face], np.nan)
at_soft_top = at_soft[:n_per_face]

# ── Compute conduction velocity directly (linear fit, AT vs x) ────────────
valid = np.isfinite(at_hard_top)
x_valid = top_x[valid]
at_valid = at_hard_top[valid]
# Linear fit: activation time (ms) = slope * x (mm) + intercept
slope_ms_per_mm, intercept = np.polyfit(x_valid, at_valid, 1)
# (mm/ms) -> (m/s): 1 mm/ms = 1 m/s exactly
cv_m_per_s = (1.0 / slope_ms_per_mm) if slope_ms_per_mm != 0 else np.nan
in_clinical_range = CLINICAL_CV_MIN_MS <= cv_m_per_s <= CLINICAL_CV_MAX_MS
print(f"  Measured CV: {cv_m_per_s:.3f} m/s  "
      f"(3D LA forward simulation range: {CLINICAL_CV_MIN_MS}-{CLINICAL_CV_MAX_MS} m/s, "
      f"in range: {in_clinical_range})")

gap = np.abs(at_hard_top - at_soft_top)
gap_valid = gap[valid]
mean_gap, max_gap = np.nanmean(gap_valid), np.nanmax(gap_valid)

# ── Figure ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200,
})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# ── A: Activation map with isochrones + propagation arrow ─────────────────
ax = axes[0]
vmin, vmax = at_valid.min(), at_valid.max()
tcf = ax.tripcolor(triang, at_hard_top, cmap="turbo", vmin=vmin, vmax=vmax,
                    shading="gouraud")
iso_lev = np.arange(np.floor(vmin / 5) * 5, vmax, 5)
if len(iso_lev) > 1:
    try:
        cs = ax.tricontour(triang, at_hard_top, levels=iso_lev,
                            colors="white", linewidths=0.6, alpha=0.85)
        ax.clabel(cs, fmt="%d ms", fontsize=6, inline=True)
    except Exception:
        pass
# Propagation-direction arrow (paced edge is x=0, wave travels +x)
arrow_y = LY_MM * 0.5
ax.annotate("", xy=(LX_MM * 0.35, arrow_y), xytext=(LX_MM * 0.08, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2,
                             mutation_scale=18))
ax.text(LX_MM * 0.05, arrow_y + LY_MM * 0.06, "propagation",
        color="white", fontsize=7.5, fontweight="bold")
cb = plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, shrink=0.85, aspect=18)
cb.set_label("Activation time (ms)")
ax.set_title("A. Activation map (isochrones, 5ms)", fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

# ── B: Conduction-velocity profile, checked against the 3D LA forward
#      simulation (patient MRI-derived mesh) ──────────────────────────────
ax = axes[1]
# Bin AT by x (mean +- std across y, at each x column) -- shows the
# genuine spread as well as the fitted line, more informative than a
# raw scatter of every node
x_unique = np.unique(x_valid)
at_mean_by_x = np.array([at_valid[x_valid == xv].mean() for xv in x_unique])
at_std_by_x = np.array([at_valid[x_valid == xv].std() for xv in x_unique])
ax.fill_between(x_unique, at_mean_by_x - at_std_by_x, at_mean_by_x + at_std_by_x,
                 color="#0072B2", alpha=0.15, label="±1 s.d. across y")
ax.plot(x_unique, at_mean_by_x, color="#0072B2", lw=1.4, label="Measured AT (mean)")
x_fit = np.array([x_unique.min(), x_unique.max()])
ax.plot(x_fit, slope_ms_per_mm * x_fit + intercept, color="#D55E00", lw=1.8,
         ls="--", label=f"Linear fit: CV = {cv_m_per_s:.3f} m/s")
# 3D LA forward simulation (patient MRI-derived mesh)'s known CV range,
# shown as a shaded reference band anchored at the same starting AT as
# the fit
t0_clinical_min = intercept + (1.0 / CLINICAL_CV_MIN_MS) * x_fit
t0_clinical_max = intercept + (1.0 / CLINICAL_CV_MAX_MS) * x_fit
ax.fill_between(x_fit, t0_clinical_max, t0_clinical_min,
                 color="#2E7D32", alpha=0.15,
                 label=f"3D LA forward simulation\n(patient MRI-derived mesh)\nCV range\n({CLINICAL_CV_MIN_MS}-{CLINICAL_CV_MAX_MS} m/s)")
ax.set_xlabel("x-distance from paced edge (mm)")
ax.set_ylabel("Activation time (ms)")
status_text = "within patient-mesh range" if in_clinical_range else "outside patient-mesh range"
status_color = "#2E7D32" if in_clinical_range else "#D55E00"
ax.set_title(f"B. Conduction velocity: {cv_m_per_s:.3f} m/s ({status_text})",
             fontweight="bold", color=status_color, fontsize=9.5)
ax.legend(frameon=False, fontsize=6.8, loc="upper left")
ax.grid(lw=0.3, alpha=0.3)

# ── C: Hard vs soft activation-time agreement ──────────────────────────
ax = axes[2]
sc = ax.scatter(at_hard_top[valid], at_soft_top[valid],
                 c=top_x[valid], cmap="viridis", s=4, alpha=0.55,
                 rasterized=True)
lims = [min(at_hard_top[valid].min(), at_soft_top[valid].min()),
        max(at_hard_top[valid].max(), at_soft_top[valid].max())]
ax.plot(lims, lims, color="#888888", lw=1.0, ls=":", label="Perfect agreement")
cb3 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85, aspect=18)
cb3.set_label("x-distance from paced edge (mm)")
ax.set_xlabel("Hard AT (ms)"); ax.set_ylabel("Soft AT (ms)")
ax.set_title(f"C. Hard vs soft AT agreement\n"
             f"(mean gap {mean_gap:.2f}ms, max {max_gap:.2f}ms)",
             fontweight="bold")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.grid(lw=0.3, alpha=0.3)

fig.suptitle(f"3D Thin-Plate Cube — Forward-Pass Physics Validation "
             f"({LX_MM}×{LY_MM}mm top face)", fontweight="bold", y=1.03, fontsize=12.5)
plt.tight_layout()
fname = os.path.join(OUT_DIR, "figure_cube_forward.png")
fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {fname}")