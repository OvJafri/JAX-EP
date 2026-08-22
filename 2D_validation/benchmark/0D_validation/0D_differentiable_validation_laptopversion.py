# -*- coding: utf-8 -*-

"""
0D (single-cell) differentiable benchmark:
JAX-EP mMS vs openCARP bench.pt

Purpose
-------
Validate the JAX-EP 0D Mitchell-Schaeffer implementation against
the openCARP bench.pt reference while retaining a differentiable
forward model suitable for gradient-based parameter learning.

Implementation
--------------
- Mitchell-Schaeffer ionic model
- Smooth sigmoid approximation of the h-gate
- Euler integration for V and h
- Current-injection stimulus (NOT voltage clamp)
- Fixed 2 ms square stimulus
- JAX lax.scan time integration
- Fully JAX-native differentiable forward model
- dt = 0.05 ms
- Same mMS parameters as benchmark2.py / manuscript

CARP reference
--------------
bench.pt --imp mMS --validate --stim-curr 80 --stim-dur 2
           --bcl 500 --numstim 1

IMPORTANT
---------
The differentiable forward model is run entirely in JAX.
NumPy/Pandas are used only AFTER the forward simulation for
loading the CARP reference, calculating validation metrics,
and plotting.
"""

import os
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==============================================================
# 0. JAX CONFIGURATION
# ==============================================================

jax.config.update("jax_enable_x64", True)


# ==============================================================
# 1. PATHS
# ==============================================================

CARP_TRACE = (
    r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects"
    r"\Jax_codes\Jax_EP\Results\benchmark\mms_trace1.dat"
)

OUT_DIR = (
    r"C:\Users\exx915\Documents\DERIstuff\Brompton"
    r"_Projects\Jax_codes\Jax_EP\Results\benchmark"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================
# 2. MITCHELL-SCHAEFFER PARAMETERS
# ==============================================================

TAU_IN    = 0.300
TAU_OUT   = 5.000
TAU_OPEN  = 120.0
TAU_CLOSE = 150.0

V_GATE = 0.13
A_CRIT = 0.13

CM = 1.0


# ==============================================================
# 3. TIME DISCRETISATION
# ==============================================================

DT    = 0.05       # ms
TOTAL = 500.0      # ms

NT = int(TOTAL / DT)

t_ms = np.arange(NT) * DT
t_j  = jnp.arange(NT, dtype=jnp.float64) * DT


# ==============================================================
# 4. STIMULUS
# ==============================================================

# Current-injection stimulus.
#
# bench.pt command:
#
#   --stim-curr 80
#   --stim-dur 2
#
# The effective normalised stimulus amplitude used in the
# previously validated JAX 0D implementation is 0.2.
#
# Keep this value if the objective is to reproduce the
# previously obtained CARP trace.

STIM_AMP = 0.2       # normalised effective stimulus amplitude
STIM_DUR = 2.0       # ms
S1_START = 1.0       # ms

stim_j = jnp.where(
    (t_j >= S1_START) &
    (t_j <  S1_START + STIM_DUR),
    jnp.float64(STIM_AMP),
    jnp.float64(0.0)
)


# ==============================================================
# 5. DIFFERENTIABLE 0D MITCHELL-SCHAEFFER SOLVER
# ==============================================================

def mms_ion_step(V, h, dt, tau_in, tau_out,
                 tau_open, tau_close,
                 v_gate, a_crit):

    # ----------------------------------------------------------
    # Smooth sigmoid approximation to the original gate switch
    #
    # This replaces:
    #
    #     where(V < V_GATE, ...)
    #
    # with a smooth function.
    # ----------------------------------------------------------

    sigmoid = jax.nn.sigmoid(
        jnp.float64(150.0) * (V - v_gate)
    )

    # Smooth transition between open and closed gate dynamics

    alpha_h = (
        (jnp.float64(1.0) - sigmoid) / tau_open
    )

    beta_h = (
        sigmoid / tau_close
    )

    dh = (
        (jnp.float64(1.0) - h) * alpha_h
        - h * beta_h
    )

    # ----------------------------------------------------------
    # Mitchell-Schaeffer currents
    # ----------------------------------------------------------

    J_in = (
        h
        * V
        * (V - a_crit)
        * (jnp.float64(1.0) - V)
        / tau_in
    )

    J_out = (
        -(jnp.float64(1.0) - h)
        * V
        / tau_out
    )

    # ----------------------------------------------------------
    # Euler update
    # ----------------------------------------------------------

    V_new = V + dt * (J_in + J_out)

    h_new = h + dt * dh

    # Numerical safety bounds.
    #
    # These are piecewise differentiable and do not introduce
    # discrete branching into the forward model.

    V_new = jnp.clip(
        V_new,
        jnp.float64(0.0),
        jnp.float64(1.0)
    )

    h_new = jnp.clip(
        h_new,
        jnp.float64(0.0),
        jnp.float64(1.0)
    )

    return V_new, h_new


# ==============================================================
# 6. DIFFERENTIABLE FORWARD MODEL
# ==============================================================

def forward_0d(params):

    (
        tau_in,
        tau_out,
        tau_open,
        tau_close,
        v_gate,
        a_crit,
    ) = params

    dt = jnp.float64(DT)

    def scan_fn(carry, stimulus):

        V, h = carry

        # ------------------------------------------------------
        # Current injection
        # ------------------------------------------------------

        I_stim = stimulus / jnp.float64(CM)

        # ------------------------------------------------------
        # Ionic currents
        # ------------------------------------------------------

        sigmoid = jax.nn.sigmoid(
            jnp.float64(150.0)
            * (V - v_gate)
        )

        alpha_h = (
            (jnp.float64(1.0) - sigmoid)
            / tau_open
        )

        beta_h = (
            sigmoid
            / tau_close
        )

        dh = (
            (jnp.float64(1.0) - h) * alpha_h
            - h * beta_h
        )

        J_in = (
            h
            * V
            * (V - a_crit)
            * (jnp.float64(1.0) - V)
            / tau_in
        )

        J_out = (
            -(jnp.float64(1.0) - h)
            * V
            / tau_out
        )

        # ------------------------------------------------------
        # Euler update
        # ------------------------------------------------------

        V_new = V + dt * (
            J_in + J_out + I_stim
        )

        h_new = h + dt * dh

        # ------------------------------------------------------
        # Numerical safety
        # ------------------------------------------------------

        V_new = jnp.clip(
            V_new,
            jnp.float64(0.0),
            jnp.float64(1.0)
        )

        h_new = jnp.clip(
            h_new,
            jnp.float64(0.0),
            jnp.float64(1.0)
        )

        return (V_new, h_new), V_new

    initial_state = (
        jnp.float64(0.0),
        jnp.float64(1.0)
    )

    (_, _), V_trace = jax.lax.scan(
        scan_fn,
        initial_state,
        stim_j
    )

    return V_trace


# JIT compiled version
forward_0d_jit = jax.jit(forward_0d)


# ==============================================================
# 7. MODEL PARAMETERS
# ==============================================================

params = jnp.array([
    TAU_IN,
    TAU_OUT,
    TAU_OPEN,
    TAU_CLOSE,
    V_GATE,
    A_CRIT,
], dtype=jnp.float64)


# ==============================================================
# 8. RUN FORWARD MODEL
# ==============================================================

print("\nRunning differentiable JAX-EP 0D solver...", flush=True)

V_jax = np.asarray(
    forward_0d_jit(params),
    dtype=np.float64
)

print(
    f"  V range: "
    f"[{V_jax.min():.6f}, {V_jax.max():.6f}]",
    flush=True
)

print(
    f"  AP peak: "
    f"{V_jax.max():.6f}",
    flush=True
)

print(
    f"  Peak time: "
    f"{t_ms[np.argmax(V_jax)]:.2f} ms",
    flush=True
)


# ==============================================================
# 9. EXPLICIT DIFFERENTIABILITY TEST
# ==============================================================

print("\nTesting automatic differentiation...", flush=True)


def scalar_loss(p):

    V = forward_0d(p)

    # Simple scalar objective
    return jnp.mean(V ** 2)


# Forward-mode Jacobian
jac_fwd = jax.jacfwd(scalar_loss)(params)

# Reverse-mode gradient
grad_rev = jax.grad(scalar_loss)(params)

print("\n  Forward-mode Jacobian:")
print(jac_fwd)

print("\n  Reverse-mode gradient:")
print(grad_rev)

print(
    "\n  Gradient finite:",
    bool(jnp.all(jnp.isfinite(grad_rev)))
)

print(
    "  Non-zero gradient:",
    bool(jnp.any(jnp.abs(grad_rev) > 0))
)


# ==============================================================
# 10. LOAD CARP REFERENCE
# ==============================================================

print(
    f"\nLoading CARP trace:\n{CARP_TRACE}",
    flush=True
)

df = pd.read_csv(
    CARP_TRACE,
    sep=r"\s+",
    header=None,
    engine="python",
    on_bad_lines="skip"
)

time_col = pd.to_numeric(
    df.iloc[:, 0],
    errors="coerce"
)

voltage_col = pd.to_numeric(
    df.iloc[:, 1],
    errors="coerce"
)

valid = ~(
    time_col.isna()
    | voltage_col.isna()
)

t_carp = time_col[valid].values
V_carp = voltage_col[valid].values

print(
    f"  Loaded {len(t_carp)} points"
)

print(
    f"  Time range: "
    f"[{t_carp.min():.1f}, {t_carp.max():.1f}] ms"
)

print(
    f"  Vm range: "
    f"[{V_carp.min():.6f}, {V_carp.max():.6f}]"
)


# ==============================================================
# 11. FIRST-BEAT CARP REFERENCE
# ==============================================================

carp_mask = t_carp <= TOTAL

t_carp_trim = t_carp[carp_mask]
V_carp_trim = V_carp[carp_mask]


# ==============================================================
# 12. ALIGN ACTIVATION
# ==============================================================

carp_act_idx = np.argmax(
    V_carp_trim > V_GATE
)

jax_act_idx = np.argmax(
    V_jax > V_GATE
)

carp_act_t = (
    t_carp_trim[carp_act_idx]
    if carp_act_idx > 0
    else 0.0
)

jax_act_t = (
    t_ms[jax_act_idx]
    if jax_act_idx > 0
    else S1_START
)

t_shift = jax_act_t - carp_act_t

print(
    f"\n  CARP activation: "
    f"{carp_act_t:.2f} ms"
)

print(
    f"  JAX activation:  "
    f"{jax_act_t:.2f} ms"
)

print(
    f"  Alignment shift:  "
    f"{t_shift:.2f} ms"
)


# ==============================================================
# 13. INTERPOLATE CARP
# ==============================================================

V_carp_interp = np.interp(
    t_ms,
    t_carp_trim + t_shift,
    V_carp_trim,
    left=0.0,
    right=0.0
)


# ==============================================================
# 14. VALIDATION METRICS
# ==============================================================

residual = V_jax - V_carp_interp

mae = float(
    np.mean(np.abs(residual))
)

rmse = float(
    np.sqrt(np.mean(residual ** 2))
)

max_err = float(
    np.max(np.abs(residual))
)

print("\nValidation metrics:")
print(
    f"  MAE:     {mae:.8f}"
)

print(
    f"  RMSE:    {rmse:.8f}"
)

print(
    f"  Max err: {max_err:.8f}"
)


# ==============================================================
# 15. FIGURE
# ==============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


C_CARP = "#1a1a1a"
C_JAX  = "#0072B2"
C_RES  = "#D55E00"


fig = plt.figure(
    figsize=(12, 8),
    facecolor="white"
)

gs = gridspec.GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.45,
    wspace=0.30,
    top=0.90,
    bottom=0.08,
    left=0.07,
    right=0.97,
    height_ratios=[1.4, 0.7]
)


def plbl(ax, label):
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left"
    )


# --------------------------------------------------------------
# A — FULL AP
# --------------------------------------------------------------

ax_a = fig.add_subplot(gs[0, :])

plbl(ax_a, "A")

ax_a.plot(
    t_ms,
    V_carp_interp,
    color=C_CARP,
    lw=2.2,
    alpha=0.85,
    label="openCARP (bench.pt reference)"
)

ax_a.plot(
    t_ms,
    V_jax,
    color=C_JAX,
    lw=1.4,
    ls="--",
    label="JAX-EP (differentiable mMS)"
)

ax_a.axvspan(
    S1_START,
    S1_START + STIM_DUR,
    alpha=0.08,
    color="gray",
    label="Current stimulus"
)

ax_a.set_xlabel("Time (ms)")
ax_a.set_ylabel("Normalised $V_m$")

ax_a.set_title(
    "0D cellular validation — differentiable "
    "Mitchell-Schaeffer model",
    fontweight="bold",
    pad=6
)

ax_a.set_xlim(0, TOTAL)
ax_a.set_ylim(-0.05, 1.1)

ax_a.legend(
    frameon=False,
    fontsize=8.5,
    loc="upper right"
)

ax_a.grid(
    lw=0.3,
    alpha=0.3
)


# --------------------------------------------------------------
# B — UPSTROKE
# --------------------------------------------------------------

ax_b = fig.add_subplot(gs[1, 0])

plbl(ax_b, "B")

zoom_end = S1_START + 30.0

ax_b.plot(
    t_ms,
    V_carp_interp,
    color=C_CARP,
    lw=2.2,
    alpha=0.85,
    label="CARP"
)

ax_b.plot(
    t_ms,
    V_jax,
    color=C_JAX,
    lw=1.4,
    ls="--",
    label="JAX-EP"
)

ax_b.axvspan(
    S1_START,
    S1_START + STIM_DUR,
    alpha=0.08,
    color="gray"
)

ax_b.set_xlim(
    S1_START - 2.0,
    zoom_end
)

ax_b.set_ylim(
    -0.02,
    1.05
)

ax_b.set_xlabel("Time (ms)")
ax_b.set_ylabel("Normalised $V_m$")

ax_b.set_title(
    "Upstroke alignment",
    fontweight="bold",
    pad=4
)

ax_b.legend(
    frameon=False,
    fontsize=8
)

ax_b.grid(
    lw=0.3,
    alpha=0.3
)


# --------------------------------------------------------------
# C — RESIDUAL
# --------------------------------------------------------------

ax_c = fig.add_subplot(gs[1, 1])

plbl(ax_c, "C")

ax_c.plot(
    t_ms,
    residual,
    color=C_RES,
    lw=1.0
)

ax_c.fill_between(
    t_ms,
    residual,
    alpha=0.25,
    color=C_RES
)

ax_c.axhline(
    0,
    color="k",
    lw=0.8,
    ls="--"
)

ax_c.set_xlabel("Time (ms)")
ax_c.set_ylabel(
    "JAX − CARP ($\\Delta V_m$)"
)

ax_c.set_title(
    "Numerical residual",
    fontweight="bold",
    pad=4
)

ax_c.set_xlim(
    0,
    TOTAL
)

ax_c.grid(
    lw=0.3,
    alpha=0.3
)

summary = (
    f"MAE:      {mae:.2e}\n"
    f"RMSE:     {rmse:.2e}\n"
    f"Max err:  {max_err:.2e}\n"
    f"dt:       {DT:.2f} ms\n"
    f"Stim:     {STIM_DUR:.1f} ms\n"
    f"Grad OK:  {bool(jnp.all(jnp.isfinite(grad_rev)))}"
)

ax_c.text(
    0.97,
    0.97,
    summary,
    transform=ax_c.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#F5F5F5",
        edgecolor="#CCCCCC",
        alpha=0.95
    )
)


# ==============================================================
# 16. SAVE
# ==============================================================

fig.suptitle(
    "JAX-EP vs openCARP | 0D single-cell benchmark | "
    "Mitchell-Schaeffer | current stimulus | "
    "differentiable JAX solver",
    fontsize=10,
    fontweight="bold",
    y=0.97
)

fname = os.path.join(
    OUT_DIR,
    "figure_0D_differentiable_JAX_vs_CARP.png"
)

fig.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ==============================================================
# 17. FINAL REPORT
# ==============================================================

print("\n==============================================")
print("0D DIFFERENTIABLE VALIDATION COMPLETE")
print("==============================================")

print(f"MAE:       {mae:.8f}")
print(f"RMSE:      {rmse:.8f}")
print(f"Max error: {max_err:.8f}")

print(
    f"Gradient finite: "
    f"{bool(jnp.all(jnp.isfinite(grad_rev)))}"
)

print(
    f"Non-zero gradient: "
    f"{bool(jnp.any(jnp.abs(grad_rev) > 0))}"
)

print(f"\nFigure saved to:")
print(fname)

print("==============================================")