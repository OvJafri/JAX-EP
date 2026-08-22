# -*- coding: utf-8 -*-
"""
JAX-EP vs openCARP | 0D single-cell benchmark
Modified Mitchell-Schaeffer (mMS) by Corrado et al ,2016 (https://doi.org/10.1016/j.mbs.2016.08.010). 

Purpose
-------
Validate the JAX-EP 0D mMS implementation against an
openCARP bench.pt reference trace while explicitly demonstrating that
the JAX forward model is differentiable.

The differentiable part is the JAX-EP model itself. CARP loading,
NumPy interpolation, plotting, and error metrics are validation/
visualisation operations and are intentionally outside the differentiable
JAX computational graph.

Model
-----
- mMS ionic model
- Sigmoid gate approximation
- Euler integration for V and h
- Differentiable JAX implementation
- dt = 0.05 ms
- 2 ms current stimulus
- Same model parameters used for the CARP benchmark

CARP reference
https://opencarp.org/documentation/examples
-------------
Generated with:

    bench.pt --imp mMS --validate --stim-curr 80 --stim-dur 2 \
              --bcl 500 --numstim 1

The CARP reference trace is loaded from the local benchmark directory.
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


# ============================================================
# 1. JAX CONFIGURATION
# ============================================================

jax.config.update("jax_enable_x64", True)


# ============================================================
# 2. PATHS
# ============================================================

# GitHub-compatible default:
# keep the CARP trace in the repository's benchmark/ directory.
#
# If repository layout is:
#
#   Jax_EP/
#       benchmark/
#           mms_trace1.dat
#       0D_differentiable_validation.py
#
# this path works directly.
#
# Otherwise, change CARP_TRACE to the location of mms_trace1.dat.

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

CARP_TRACE = os.path.join(
    REPO_DIR,
    "mms_trace1.dat",
)

OUT_DIR = REPO_DIR


# ============================================================
# 3. MODIFIED MITCHELL-SCHAEFFER PARAMETERS
# ============================================================

TAU_IN = 0.300
TAU_OUT = 5.000
TAU_OPEN = 120.0
TAU_CLOSE = 150.0

V_GATE = 0.13
A_CRIT = 0.13

CM = 1.0
N_ION = 2


# ============================================================
# 4. TIME AND STIMULUS
# ============================================================

DT = 0.05          # ms
TOTAL = 500.0      # ms

NT = int(TOTAL / DT)

t_ms = np.arange(NT) * DT

# Current stimulus corresponding to the CARP benchmark:
# bench.pt --stim-curr 80 --stim-dur 2
STIM_AMP = 80.0    # µA/cm²
STIM_DUR = 2.0     # ms
S1_START = 10.0    # ms

sv = (
    (t_ms >= S1_START)
    & (t_ms < S1_START + STIM_DUR)
)

sv_j = jnp.array(sv)

# Sub-stepping used by the 0D ionic update.
dt_sub = DT / (2 * N_ION)


# ============================================================
# 5. DIFFERENTIABLE JAX 0D FORWARD MODEL
# ============================================================

def ion_step(V, h):
    """
    One differentiable Mitchell-Schaeffer ionic update.

    All operations in this function are JAX operations so that the
    model can be differentiated with respect to its parameters.
    """

    sw = jax.nn.sigmoid(
        jnp.float64(150.0)
        * (V - jnp.float64(V_GATE))
    )

    dh = (
        ((jnp.float64(1.0) - h) / jnp.float64(TAU_OPEN))
        * (jnp.float64(1.0) - sw)
        -
        (h / jnp.float64(TAU_CLOSE))
        * sw
    )

    J_in = (
        h
        * V
        * (V - jnp.float64(A_CRIT))
        * (jnp.float64(1.0) - V)
        / jnp.float64(TAU_IN)
    )

    J_out = (
        -(jnp.float64(1.0) - h)
        * (V / jnp.float64(TAU_OUT))
    )

    Vn = jnp.clip(
        V + jnp.float64(dt_sub) * (J_in + J_out),
        jnp.float64(0.0),
        jnp.float64(1.0),
    )

    hn = jnp.clip(
        h + jnp.float64(dt_sub) * dh,
        jnp.float64(0.0),
        jnp.float64(1.0),
    )

    return Vn, hn


@jax.jit
def run_0d(params):
    """
    Run the differentiable JAX 0D model.

    Parameters
    ----------
    params : array-like, shape (6,)
        [tau_in, tau_out, tau_open, tau_close, v_gate, a_crit]

    Returns
    -------
    V_trace : jax.Array
        Simulated membrane-voltage trace.
    """

    tau_in, tau_out, tau_open, tau_close, v_gate, a_crit = params

    def _ion(V, h):
        sw = jax.nn.sigmoid(
            jnp.float64(150.0)
            * (V - v_gate)
        )

        dh = (
            ((jnp.float64(1.0) - h) / tau_open)
            * (jnp.float64(1.0) - sw)
            -
            (h / tau_close) * sw
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
            * (V / tau_out)
        )

        Vn = jnp.clip(
            V + jnp.float64(dt_sub) * (J_in + J_out),
            jnp.float64(0.0),
            jnp.float64(1.0),
        )

        hn = jnp.clip(
            h + jnp.float64(dt_sub) * dh,
            jnp.float64(0.0),
            jnp.float64(1.0),
        )

        return Vn, hn

    def scan_fn(carry, sv_t):
        V, h = carry

        for _ in range(N_ION):
            V, h = _ion(V, h)

        # Current stimulus represented as a voltage clamp, matching
        # the CARP benchmark convention used for this validation.
        V = jnp.where(
            sv_t,
            jnp.float64(1.0),
            V,
        )

        for _ in range(N_ION):
            V, h = _ion(V, h)

        return (V, h), V

    V0 = jnp.float64(0.0)
    h0 = jnp.float64(1.0)

    (_, _), V_trace = jax.lax.scan(
        scan_fn,
        (V0, h0),
        sv_j,
    )

    return V_trace


# ============================================================
# 6. BASELINE PARAMETERS
# ============================================================

PARAMS = jnp.array(
    [
        TAU_IN,
        TAU_OUT,
        TAU_OPEN,
        TAU_CLOSE,
        V_GATE,
        A_CRIT,
    ],
    dtype=jnp.float64,
)


# ============================================================
# 7. RUN FORWARD MODEL
# ============================================================

print("\nRunning differentiable JAX-EP 0D solver...", flush=True)

V_jax = np.array(
    run_0d(PARAMS),
    dtype=np.float64,
)

print(
    f"  V range: [{V_jax.min():.6f}, {V_jax.max():.6f}]",
    flush=True,
)

peak_idx = np.argmax(V_jax)

print(
    f"  AP peak: {V_jax.max():.6f}",
    flush=True,
)

print(
    f"  Peak time: {t_ms[peak_idx]:.2f} ms",
    flush=True,
)


# ============================================================
# 8. AUTODIFF TEST
# ============================================================

print("\nTesting automatic differentiation...", flush=True)


# Use the final membrane voltage as a scalar differentiable
# quantity for the parameter-gradient demonstration.
def final_voltage(params):
    return run_0d(params)[-1]


jac_forward = np.array(
    jax.jacfwd(final_voltage)(PARAMS),
    dtype=np.float64,
)

grad_reverse = np.array(
    jax.grad(final_voltage)(PARAMS),
    dtype=np.float64,
)

gradient_finite = bool(
    np.all(np.isfinite(jac_forward))
    and np.all(np.isfinite(grad_reverse))
)

gradient_nonzero = bool(
    np.any(np.abs(jac_forward) > 0.0)
    and np.any(np.abs(grad_reverse) > 0.0)
)

print("\n  Forward-mode Jacobian:")
print(jac_forward)

print("\n  Reverse-mode gradient:")
print(grad_reverse)

print(
    f"\n  Gradient finite: {gradient_finite}"
)

print(
    f"  Non-zero gradient: {gradient_nonzero}"
)


# ============================================================
# 9. LOAD openCARP REFERENCE
# ============================================================

print("\nLoading CARP trace:")
print(CARP_TRACE)

if not os.path.isfile(CARP_TRACE):
    raise FileNotFoundError(
        "\nCARP reference trace not found:\n"
        f"{CARP_TRACE}\n\n"
        "Place mms_trace1.dat in the benchmark/ directory "
        "or update CARP_TRACE in this script."
    )

df = pd.read_csv(
    CARP_TRACE,
    sep=r"\s+",
    header=None,
    engine="python",
    on_bad_lines="skip",
)

time_col = pd.to_numeric(
    df.iloc[:, 0],
    errors="coerce",
)

voltage_col = pd.to_numeric(
    df.iloc[:, 1],
    errors="coerce",
)

valid = ~(
    time_col.isna()
    | voltage_col.isna()
)

t_carp = time_col[valid].values
V_carp = voltage_col[valid].values

print(f"  Loaded {len(t_carp)} points", flush=True)
print(
    f"  Time range: [{t_carp.min():.1f}, "
    f"{t_carp.max():.1f}] ms",
    flush=True,
)
print(
    f"  Vm range:   [{V_carp.min():.6f}, "
    f"{V_carp.max():.6f}]",
    flush=True,
)


# ============================================================
# 10. ALIGN CARP AND JAX TRACES
# ============================================================

carp_mask = t_carp <= TOTAL

t_carp_trim = t_carp[carp_mask]
V_carp_trim = V_carp[carp_mask]

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
    f"\n  CARP activation: {carp_act_t:.2f} ms"
)

print(
    f"  JAX activation:  {jax_act_t:.2f} ms"
)

print(
    f"  Alignment shift: {t_shift:.2f} ms"
)


V_carp_interp = np.interp(
    t_ms,
    t_carp_trim + t_shift,
    V_carp_trim,
    left=0.0,
    right=0.0,
)


# ============================================================
# 11. VALIDATION METRICS
# ============================================================

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
print(f"  MAE:     {mae:.8f}")
print(f"  RMSE:    {rmse:.8f}")
print(f"  Max err: {max_err:.8f}")


# ============================================================
# 12. FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_CARP = "#1a1a1a"
C_JAX = "#0072B2"
C_RES = "#D55E00"


fig = plt.figure(
    figsize=(12, 8),
    facecolor="white",
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
    height_ratios=[1.4, 0.7],
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
        ha="left",
    )


# ------------------------------------------------------------
# Panel A: full AP trace
# ------------------------------------------------------------

ax_a = fig.add_subplot(gs[0, :])

plbl(ax_a, "A")

ax_a.plot(
    t_ms,
    V_carp_interp,
    color=C_CARP,
    lw=2.2,
    alpha=0.85,
    label="openCARP (bench.pt reference)",
    zorder=3,
)

ax_a.plot(
    t_ms,
    V_jax,
    color=C_JAX,
    lw=1.4,
    ls="--",
    label="JAX-EP (differentiable mMS)",
    zorder=4,
)

ax_a.axvspan(
    S1_START,
    S1_START + STIM_DUR,
    alpha=0.08,
    color="gray",
    label="Current stimulus",
)

ax_a.set_xlabel("Time (ms)", fontsize=10)
ax_a.set_ylabel("Normalised $V_m$", fontsize=10)

ax_a.set_title(
    "0D cellular validation — differentiable "
    "Mitchell-Schaeffer model",
    fontweight="bold",
    pad=6,
)

ax_a.set_xlim(0, TOTAL)
ax_a.set_ylim(-0.05, 1.1)

ax_a.legend(
    frameon=False,
    fontsize=8.5,
    loc="upper right",
)

ax_a.grid(
    lw=0.3,
    alpha=0.3,
)


# ------------------------------------------------------------
# Panel B: upstroke zoom
# ------------------------------------------------------------

ax_b = fig.add_subplot(gs[1, 0])

plbl(ax_b, "B")

zoom_end = S1_START + 30.0

ax_b.plot(
    t_ms,
    V_carp_interp,
    color=C_CARP,
    lw=2.2,
    alpha=0.85,
    label="CARP",
)

ax_b.plot(
    t_ms,
    V_jax,
    color=C_JAX,
    lw=1.4,
    ls="--",
    label="JAX-EP",
)

ax_b.axvspan(
    S1_START,
    S1_START + STIM_DUR,
    alpha=0.08,
    color="gray",
)

ax_b.set_xlim(
    S1_START - 2.0,
    zoom_end,
)

ax_b.set_ylim(
    -0.02,
    1.05,
)

ax_b.set_xlabel(
    "Time (ms)",
    fontsize=10,
)

ax_b.set_ylabel(
    "Normalised $V_m$",
    fontsize=10,
)

ax_b.set_title(
    "Upstroke alignment",
    fontweight="bold",
    pad=4,
)

ax_b.legend(
    frameon=False,
    fontsize=8,
)

ax_b.grid(
    lw=0.3,
    alpha=0.3,
)


# ------------------------------------------------------------
# Panel C: residual
# ------------------------------------------------------------

ax_c = fig.add_subplot(gs[1, 1])

plbl(ax_c, "C")

ax_c.plot(
    t_ms,
    residual,
    color=C_RES,
    lw=1.0,
)

ax_c.fill_between(
    t_ms,
    residual,
    alpha=0.25,
    color=C_RES,
)

ax_c.axhline(
    0,
    color="k",
    lw=0.8,
    ls="--",
)

ax_c.set_xlabel(
    "Time (ms)",
    fontsize=10,
)

ax_c.set_ylabel(
    "JAX − CARP ($\\Delta V_m$)",
    fontsize=10,
)

ax_c.set_title(
    "Numerical residual",
    fontweight="bold",
    pad=4,
)

ax_c.set_xlim(
    0,
    TOTAL,
)

ax_c.grid(
    lw=0.3,
    alpha=0.3,
)

summary = (
    f"MAE:     {mae:.2e}\n"
    f"RMSE:    {rmse:.2e}\n"
    f"Max err: {max_err:.2e}\n"
    f"dt:      {DT:.2f} ms\n"
    f"Stim:    {STIM_DUR:.1f} ms\n"
    f"Grad OK: {gradient_finite and gradient_nonzero}"
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
        alpha=0.95,
    ),
)


fig.suptitle(
    "JAX-EP vs openCARP | 0D single-cell benchmark | "
    "Mitchell-Schaeffer | current stimulus | "
    "differentiable JAX solver",
    fontsize=10,
    fontweight="bold",
    y=0.97,
)


# ============================================================
# 13. SAVE FIGURE
# ============================================================

figure_path = os.path.join(
    OUT_DIR,
    "figure_0D_differentiable_JAX_vs_CARP.png",
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("0D DIFFERENTIABLE VALIDATION COMPLETE")
print("==============================================")
print(f"MAE:       {mae:.8f}")
print(f"RMSE:      {rmse:.8f}")
print(f"Max error: {max_err:.8f}")
print(f"Gradient finite: {gradient_finite}")
print(f"Non-zero gradient: {gradient_nonzero}")
print("\nFigure saved to:")
print(figure_path)
print("==============================================")
