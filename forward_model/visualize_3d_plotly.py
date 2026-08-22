# -*- coding: utf-8 -*-
"""
visualize_3d_plotly.py
==========================
Genuine, interactive 3D visualization of the full thin-plate cube's
activation-time map, using Plotly's Mesh3d -- which correctly
handles depth/occlusion, so the FULL shell (top face + bottom face +
side walls) can be shown at once without the projection-overlap
issue a flat 2D matplotlib projection would have. Paired with the
CPU-vs-GPU performance table (2D Tissue-Sample Forward Performance,
NVIDIA Tesla P100), so the wavefront visualization and the GPU
speedup result are shown together in one figure.

Loads the output of run_forward_only.py (cube_forward_only_atmap.npz)
and renders an interactive HTML file you can open and rotate/zoom in
a browser, styled as a clean, modern, dark-theme scientific
visualization -- the wavefront glows across the plate, the pacing
site is marked distinctly, and the colorbar is sized proportionately
(fixed: it previously defaulted to a full-height bar that looked far
too tall against this thin plate's own aspect ratio).
"""
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", "./outputs")
IN_PATH = os.path.join(OUT_DIR, "cube_forward_only_atmap.npz")

# ── CPU-vs-GPU performance table (2D Tissue-Sample Forward Performance,
#    NVIDIA Tesla P100) -- edit these if you re-run the benchmark
#    yourself and get different numbers ─────────────────────────────────
PERF_CPU_TIME_S = 57.732
PERF_GPU_TIME_S = 1.342
PERF_SPEEDUP = PERF_CPU_TIME_S / PERF_GPU_TIME_S
PERF_ACTIVATED_NODES = 18432
PERF_AT_RANGE = "13.10–54.40 ms"
PERF_MAX_AT_DIFF_MS = 0.000

print(f"Loading {IN_PATH} ...")
data = np.load(IN_PATH)
Verts = data["Verts"]
Elems = data["Elems"]
at_hard = data["at_hard"]
activated = data["activated"]
print(f"  {len(Verts):,} nodes  {len(Elems):,} elements")

# Color by activation time; mark never-activated nodes distinctly
# rather than silently including them at some arbitrary value
at_plot = np.where(activated, at_hard, np.nan)
n_never_activated = int((~activated).sum())
if n_never_activated > 0:
    print(f"  NOTE: {n_never_activated} node(s) never activated -- "
          f"shown as gaps (NaN) in the color map, not a fixed color.")

vmin = float(np.nanmin(at_plot))
vmax = float(np.nanmax(at_plot))

# Identify the paced nodes directly from the data (earliest-activating
# band), so the wavefront's own origin is shown explicitly rather than
# left for the viewer to infer from color alone
activated_at = at_plot[activated]
paced_threshold = np.nanpercentile(activated_at, 2)  # earliest ~2% of activation times
paced_mask = activated & (at_plot <= paced_threshold)
paced_pts = Verts[paced_mask]

# ── Dark, modern colorscale: deep indigo (unactivated/earliest) through
#    cyan/teal to a hot amber/white leading edge -- reads clearly as a
#    "wavefront" against a dark background, not a generic rainbow ──────
WAVEFRONT_COLORSCALE = [
    [0.00, "#1a0a3e"],
    [0.15, "#3d1a7a"],
    [0.32, "#2563eb"],
    [0.50, "#06b6d4"],
    [0.68, "#34d399"],
    [0.84, "#fbbf24"],
    [1.00, "#fef3c7"],
]

# ── Layout: 3D wavefront (left, wide) + performance table (right) ────────
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.68, 0.32],
    specs=[[{"type": "scene"}, {"type": "table"}]],
    horizontal_spacing=0.02,
)

# Main activation-map mesh
fig.add_trace(go.Mesh3d(
    x=Verts[:, 0], y=Verts[:, 1], z=Verts[:, 2],
    i=Elems[:, 0], j=Elems[:, 1], k=Elems[:, 2],
    intensity=at_plot,
    colorscale=WAVEFRONT_COLORSCALE,
    cmin=vmin, cmax=vmax,
    colorbar=dict(
        title=dict(text="Activation<br>time (ms)", font=dict(color="#e5e7eb", size=13)),
        len=0.5, thickness=16, x=0.62, y=0.5,
        tickfont=dict(color="#e5e7eb", size=11),
        outlinewidth=0, bgcolor="rgba(0,0,0,0)",
    ),
    showscale=True,
    flatshading=False,
    lighting=dict(ambient=0.45, diffuse=0.75, specular=0.75, roughness=0.35,
                   fresnel=0.15),
    lightposition=dict(x=100, y=200, z=300),
    hovertemplate="AT: %{intensity:.1f} ms<br>x: %{x:.1f}mm  y: %{y:.1f}mm<extra></extra>",
    name="Activation map",
), row=1, col=1)

# Pacing-site marker -- shows the wave's genuine origin explicitly
if len(paced_pts) > 0:
    fig.add_trace(go.Scatter3d(
        x=paced_pts[:, 0], y=paced_pts[:, 1], z=paced_pts[:, 2],
        mode="markers",
        marker=dict(size=2.2, color="#ffffff", opacity=0.9,
                     line=dict(width=0)),
        name="Pacing site",
        hovertemplate="Pacing site<extra></extra>",
    ), row=1, col=1)

# ── Performance table: CPU vs GPU, matching the manuscript's own
#    2D Tissue-Sample Forward Performance table exactly ─────────────────
table_header = ["<b>Metric</b>", "<b>CPU</b>", "<b>GPU</b>", "<b>GPU Speed-up</b>"]
table_rows = [
    ["<b>Forward simulation</b>", f"{PERF_CPU_TIME_S:.3f} s",
     f"<b>{PERF_GPU_TIME_S:.3f} s</b>", f"<b>{PERF_SPEEDUP:.1f}×</b>"],
    ["Activated nodes", f"{PERF_ACTIVATED_NODES:,}", f"{PERF_ACTIVATED_NODES:,}", "—"],
    ["Activation-time range", PERF_AT_RANGE, PERF_AT_RANGE, "—"],
    ["Activation map agreement", "—", "<b>Identical</b>",
     f"<b>{PERF_MAX_AT_DIFF_MS:.3f} ms</b> max diff"],
]
cols = list(zip(*table_rows))  # transpose rows -> columns for go.Table

fig.add_trace(go.Table(
    header=dict(
        values=table_header,
        fill_color="#1f2937",
        font=dict(color="#f3f4f6", size=13),
        align="left",
        height=32,
        line=dict(color="#374151", width=1),
    ),
    cells=dict(
        values=cols,
        fill_color=[["#161a23", "#1a1e29", "#161a23", "#1a1e29"]],
        font=dict(color="#e5e7eb", size=12),
        align="left",
        height=30,
        line=dict(color="#2a2f3a", width=1),
    ),
    columnwidth=[34, 22, 22, 30],
), row=1, col=2)

fig.update_layout(
    title=dict(
        text=(f"<b>3D Thin-Plate Cube — Activation Wavefront & GPU Speedup</b>"
              f"<br><span style='font-size:13px;color:#9ca3af'>"
              f"{len(Verts):,} nodes · full shell · AT range "
              f"{vmin:.1f}–{vmax:.1f}ms · {PERF_SPEEDUP:.0f}× GPU speedup "
              f"(NVIDIA Tesla P100)</span>"),
        font=dict(color="#f3f4f6", size=21, family="DejaVu Sans, Arial"),
        x=0.02, xanchor="left", y=0.97,
    ),
    scene=dict(
        xaxis=dict(title="x (mm)", color="#9ca3af", gridcolor="#374151",
                    backgroundcolor="#0f1117", zerolinecolor="#374151"),
        yaxis=dict(title="y (mm)", color="#9ca3af", gridcolor="#374151",
                    backgroundcolor="#0f1117", zerolinecolor="#374151"),
        zaxis=dict(title="z (mm)", color="#9ca3af", gridcolor="#374151",
                    backgroundcolor="#0f1117", zerolinecolor="#374151"),
        aspectmode="data",   # preserves true relative proportions
                              # (critical here: the plate is ~19mm in-
                              # plane but only ~1.5mm thick -- without
                              # aspectmode="data" plotly would default
                              # to a cube-shaped bounding box and
                              # grossly exaggerate the thickness)
        camera=dict(
            eye=dict(x=1.35, y=-1.55, z=1.1),
            up=dict(x=0, y=0, z=1),
        ),
        domain=dict(x=[0.0, 0.62]),
    ),
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    legend=dict(font=dict(color="#e5e7eb"), bgcolor="rgba(0,0,0,0)",
                 x=0.02, y=0.06),
    width=1350, height=750,
    margin=dict(l=0, r=0, t=100, b=0),
)

out_html = os.path.join(OUT_DIR, "cube_3d_activation.html")
fig.write_html(out_html)
print(f"\nSaved interactive visualization: {out_html}")
print("Open this file in a web browser to rotate/zoom/inspect.")