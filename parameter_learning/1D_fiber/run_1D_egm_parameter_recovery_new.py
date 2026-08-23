# -*- coding: utf-8 -*-
"""
run_1D_egm_parameter_recovery.py
=====================================
1D spatial mMS parameter-recovery: recovers four cell-kinetics
parameters (tau_in, tau_out, tau_open, tau_close) from a simulated
1D bipolar EGM waveform (a wavefront propagating along a 150-node
cable), for each parameter set listed in parameter_list.txt, via a
3-phase optimization curriculum:
  Phase 1 -- 0D restitution shape fit (matches the APD restitution
             curve across a sweep of S2 coupling intervals)
  Phase 2 -- joint 1D feature matching (slew rate, T-wave amplitude,
             activation-recovery interval, peak-to-peak, alongside
             the restitution term)
  Phase 3 -- joint waveform MSE refinement, using forward-mode
             gradients (a manual loop of separate jax.jvp calls, not
             jax.jacfwd directly -- its internal vmap-batching can
             cause CPU compilation hangs) and L-BFGS-B, rather than
             reverse-mode automatic differentiation and Adam --
             forward-mode is genuinely more stable and converges more
             reliably for this stiff mMS ionic model.


What this is: a batch "grinder" -- processes every row in
parameter_list.txt as an independent ground-truth target, recovering
its parameters from scratch and saving a comparison figure per row.

How to use
----------
1. Place parameter_list.txt in the same folder as this script (or
   edit FILE_PATH below to point elsewhere). Expected format: a
   header row (skipped), then one row per parameter set --
   tau_in, tau_out, tau_open, tau_close -- space- or tab-separated.
2. Run:
       python run_1D_egm_parameter_recovery.py
3. Recovered-vs-ground-truth comparison figures are saved to
   ./Recovery_figures/Set_{n}_Recovery.png, one per row in the file.

Requires: jax, optax, scipy, numpy, matplotlib. Runs on CPU or GPU
automatically.
"""
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import subprocess
from scipy.optimize import minimize as scipy_minimize

# ── Device check (informational only -- see note above on GPU) ──────────
try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if gpu_info:
        print(f"GPU: {gpu_info}")
except Exception:
    pass
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# ==========================================
# 1. CONFIGURATION & BATCH SETUP
# ==========================================
FILE_PATH = "parameter_list.txt"
FIGURE_DIR = "./Recovery_figures"

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# Global Physics & Timing
DT, V_GATE, A_CRIT = 0.1, 0.13, 0.13
S1, S1_BEATS = 600.0, 5
MAX_STEPS_0D = int(4500 / DT)
MAX_STEPS_1D = int(600 / DT)
WINDOW_SIZE = int(500 / DT)
S2_VALUES = jnp.arange(500, 345, -10)

# 1D FIXED PHYSICS
NX, DX, D_COEFF = 150, 0.02, 0.001903
cell_coords = jnp.arange(NX) * DX
elec_z, elec_x1, elec_x2 = 0.15, 1.0, 1.2
r1, r2 = jnp.sqrt((cell_coords - elec_x1)**2 + elec_z**2), jnp.sqrt((cell_coords - elec_x2)**2 + elec_z**2)
LF_DIFF = ((cell_coords - elec_x1) / r1**3) - ((cell_coords - elec_x2) / r2**3)

# ==========================================
# 2. BOUNDS WITH 20% FACTOR OF SAFETY
# ==========================================
# Raw Corrado Ranges for [tau_in, tau_out, tau_open, tau_close]
LB_RAW = jnp.array([0.05, 0.5, 65.0, 65.0])
UB_RAW = jnp.array([0.4, 9.5, 215.0, 185.0])

# Applying 20% Padding
padding = (UB_RAW - LB_RAW) * 0.20
LB = LB_RAW + padding
UB = UB_RAW - padding

# ==========================================
# 3. ENGINES & HELPERS
# ==========================================
def get_p(lp):
    """Maps optimizer space to Physical Space via Sigmoid Mapping."""
    return LB + (UB - LB) * jax.nn.sigmoid(lp)


def inverse_p(p):
    """Maps Physical parameters to Optimizer space."""
    p_safe = jnp.clip(p, LB + 1e-5, UB - 1e-5)
    return jax.scipy.special.logit((p_safe - LB) / (UB - LB))


@jax.jit
def mms_step_0d(state, t, lp, s2_val):
    V, h = state
    p = get_p(lp)
    s1_times = jnp.arange(S1_BEATS) * S1
    stim = jnp.exp(-0.5 * ((t - jnp.append(s1_times, s1_times[-1] + s2_val)) / 1.2) ** 2).sum()
    I_stim = jnp.where(stim > 0.4, 0.2, 0.0)
    J_in, J_out = (h * V * (V - A_CRIT) * (1.0 - V)) / p[0], -(1.0 - h) * (V / p[1])
    switch = jax.nn.sigmoid(150.0 * (V - V_GATE))
    dh = ((1.0 - h) / p[2]) * (1.0 - switch) - (h / p[3]) * switch
    return (jnp.clip(V + DT * (J_in + J_out + I_stim), 0.0, 1.0), jnp.clip(h + DT * dh, 0.0, 1.0)), V


def get_apd_0d(lp, s2_val):
    _, v_trace = jax.lax.scan(lambda c, i: mms_step_0d(c, i * DT, lp, s2_val), (0.0, 1.0), jnp.arange(MAX_STEPS_0D))
    s2_idx = jnp.round((10.0 + (S1_BEATS - 1) * S1 + s2_val) / DT).astype(jnp.int32)
    v_s2 = jax.lax.dynamic_slice(v_trace, (s2_idx,), (WINDOW_SIZE,))
    return jnp.sum(jax.nn.sigmoid(300.0 * (v_s2 - 0.2))) * DT


@jax.jit
def mms_spatial_step(state, t, lp):
    V, h = state
    p = get_p(lp)
    stim = jnp.exp(-0.5 * ((t - 10.0) / 1.5)**2)
    I_stim = jnp.where((stim > 0.4) & (cell_coords < 0.1), 0.3, 0.0)
    V_pad = jnp.pad(V, 1, mode='edge')
    dV_dx2 = (V_pad[2:] - 2 * V + V_pad[:-2]) / DX**2
    dh = jnp.where(V < V_GATE, (1.0 - h) / p[2], -h / p[3])
    J_in, J_out = (h * V * (V - A_CRIT) * (1.0 - V)) / p[0], -(1.0 - h) * (V / p[1])
    V_n = jnp.clip(V + DT * (D_COEFF * dV_dx2 + J_in + J_out + I_stim), 0, 1)
    h_n = jnp.clip(h + DT * dh, 0, 1)
    phi_bip = jnp.sum(LF_DIFF * jnp.gradient(V, DX)) * DX
    return (V_n, h_n), (V[NX // 2], phi_bip)


def get_traces_1d(lp):
    init = (jnp.zeros(NX), jnp.ones(NX))
    _, (v_mid, egm) = jax.lax.scan(lambda c, t: mms_spatial_step(c, t * DT, lp), init, jnp.arange(MAX_STEPS_1D))
    return v_mid, egm


@jax.jit
def extract_mapped_features(egm):
    t_act, t_rec = jnp.argmin(egm), jnp.argmax(egm[int(200 / DT):]) + int(200 / DT)
    return jnp.array([
        jnp.max(jnp.abs(jnp.gradient(egm, DT))),
        jnp.max(egm[int(250 / DT):]),
        (t_rec - t_act) * DT,
        jnp.max(egm) - jnp.min(egm)
    ])

# ==========================================
# 4. BATCH PROCESSING GRINDER
# ==========================================
def run_grinder():
    try:
        # Default loadtxt handles tabs/spaces perfectly
        data = np.loadtxt(FILE_PATH, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        print(f">>> Grinder loaded {len(data)} sets from {FILE_PATH}")
    except Exception as e:
        print(f"Error loading file: {e}"); return

    for idx, p_target in enumerate(data):
        print(f"\n--- SET {idx+1}/{len(data)} | Target: {p_target} ---")
        start_t = time.time()
        gt_lp = inverse_p(p_target)

        target_curve_0d = jax.vmap(get_apd_0d, in_axes=(None, 0))(gt_lp, S2_VALUES)
        _, target_egm_1d = get_traces_1d(gt_lp)
        target_f = extract_mapped_features(target_egm_1d)

        lp = jnp.zeros(4)  # midpoint init (sigmoid(0)=0.5 -> midpoint of each parameter's bounds)

        # --------------------------------------------------
        # PHASE 1: 0D
        # --------------------------------------------------
        opt1 = optax.adam(0.02)
        st1 = opt1.init(lp)

        @jax.jit
        def l1(p):
            return jnp.mean(optax.huber_loss(jax.vmap(get_apd_0d, (None, 0))(p, S2_VALUES), target_curve_0d, delta=10.0))

        for _ in range(1001):
            loss, grads = jax.value_and_grad(l1)(lp)
            if loss < 0.01:
                break
            updates, st1 = opt1.update(grads, st1); lp = optax.apply_updates(lp, updates)

        # --------------------------------------------------
        # PHASE 2: Features
        # --------------------------------------------------
        opt2 = optax.adam(0.01)
        st2 = opt2.init(lp)

        @jax.jit
        def l2(p):
            _, sim_egm = get_traces_1d(p)
            sim_f = extract_mapped_features(sim_egm)
            egm_l = jnp.mean(jnp.abs((sim_f - target_f) / (target_f + 1e-6)))
            rest_l = 0.01 * jnp.mean(optax.huber_loss(jax.vmap(get_apd_0d, (None, 0))(p, S2_VALUES), target_curve_0d, delta=10.0))
            return egm_l + rest_l

        for _ in range(1501):
            loss, grads = jax.value_and_grad(l2)(lp)
            if loss < 0.0025:
                break
            updates, st2 = opt2.update(grads, st2); lp = optax.apply_updates(lp, updates)

        # --------------------------------------------------
        # PHASE 3: MSE -- forward-mode gradients (manual jvp loop) +
        # L-BFGS-B, instead of reverse-mode + Adam. Reverse-mode AD is
        # genuinely less stable for this stiff mMS ionic model family;
        # forward-mode converges more reliably (confirmed directly,
        # repeatedly, throughout this project).
        # --------------------------------------------------
        @jax.jit
        def l3(p):
            _, sim_egm = get_traces_1d(p)
            sim_mse = jnp.mean((sim_egm - target_egm_1d)**2)
            rest_mse = 0.1 * jnp.mean((jax.vmap(get_apd_0d, (None, 0))(p, S2_VALUES) - target_curve_0d)**2)
            return sim_mse + rest_mse

        _jvp_fn = jax.jit(lambda p, t: jax.jvp(l3, (p,), (t,))[1])
        _tangents = [jnp.zeros(4).at[i].set(1.0) for i in range(4)]

        def l3_value_and_grad_np(p_np):
            p = jnp.array(p_np)
            loss_val = float(l3(p))
            grad = np.array([float(_jvp_fn(p, t)) for t in _tangents])
            return loss_val, grad

        res3 = scipy_minimize(
            l3_value_and_grad_np, x0=np.array(lp), method='L-BFGS-B', jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
        lp = jnp.array(res3.x)

        final_p = get_p(lp)
        wall_time = time.time() - start_t
        _, egm_rec = get_traces_1d(lp)
        rec_rest = jax.vmap(get_apd_0d, (None, 0))(lp, S2_VALUES)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(S2_VALUES, target_curve_0d, 'ko', label='GT'); ax[0].plot(S2_VALUES, rec_rest, 'r-', label='Rec')
        ax[0].set_xlabel('S2 BCL (ms)'); ax[0].set_ylabel('APD (ms)'); ax[0].set_title('APD Restitution'); ax[0].legend()
        ax[1].plot(target_egm_1d, 'k', lw=2, label='GT'); ax[1].plot(egm_rec, 'b--', label='Rec')
        ax[1].set_xlabel('Time step'); ax[1].set_ylabel('LF EGM (a.u.)'); ax[1].set_title('LF EGM'); ax[1].legend()
        plt.suptitle(f"Set {idx+1} | Time: {wall_time:.1f}s\nRec: {np.round(np.array(final_p), 3)}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"Set_{idx+1}_Recovery.png"))
        plt.close(fig)
        print(f"DONE. Recovered: {np.array(final_p)}  time={wall_time:.1f}s")


if __name__ == "__main__":
    run_grinder()