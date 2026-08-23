# -*- coding: utf-8 -*-
"""
run_0D_parameter_recovery.py
=================================
0D modified Mitchell-Schaeffer (mMS) single-cell parameter-recovery
example: blind, gradient-based recovery of four cell-kinetics
parameters (tau_in, tau_out, tau_open, tau_close) from a restitution
curve, using the same ionic solution scheme as the full 3D
patient-specific LA patch model (identical mMS ionic kinetics and
forward-Euler update), the same loss structure, and the same
gradient-based optimization approach as the full 3D
parameter-learning pipeline -- shown here at the simplest possible
scale (a single, isolated 0D cell), so the underlying method is easy
to read end-to-end.


Usage
-----
    python run_0D_parameter_recovery.py

Requires: jax, optax, numpy. Runs on CPU or GPU automatically
(whichever JAX detects) -- no platform-specific setup needed.
"""
import os
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
import optax

# ── Device check -- works on any GPU platform with nvidia-smi
#    available, not tied to any specific one ─────────────────────────────
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

OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1. Configuration & ground truth
# -----------------------------
V_GATE, A_CRIT, DT = 0.13, 0.13, 0.1
S1, S1_BEATS = 600.0, 6
MAX_STEPS = int(4500 / DT)
NUM_ITERS = 200

# S2 range: 500ms down to 300ms ONLY, 10ms spacing (21 points) --
# does NOT extend below 300ms.
s2_test = jnp.arange(500, 295, -10.0)
param_true = jnp.array([0.30, 6.0, 120.0, 150.0])  # single set

# ── Bounds -- SAME as parameter_learning_patch_geometry.py's own
#    LB5/UB5 for this mMS model family (minus G_IL, not used here).
#    Genuinely wide, not tight: e.g. tau_open spans 80-215 around a
#    true value of 120 -- these are guardrails against completely
#    wild excursions during optimization, not close constraints. ───────
LB4 = jnp.array([0.01, 0.50, 80.0, 80.0])
UB4 = jnp.array([0.40, 9.50, 215.0, 185.0])


def get_p4(lp):
    return LB4 + (UB4 - LB4) * jax.nn.sigmoid(lp)


def inv_p4(p):
    ps = jnp.clip(jnp.asarray(p), LB4 + 1e-6, UB4 - 1e-6)
    return jax.scipy.special.logit((ps - LB4) / (UB4 - LB4))


# -----------------------------
# 2. Stabilized mMS step
# -----------------------------
@jax.jit
def mms_step_stabilized(state, t, log_params, s2_val):
    V, h = state
    p = get_p4(log_params)

    s1_times = jnp.arange(S1_BEATS) * S1
    s2_time = s1_times[-1] + s2_val
    stim = jnp.exp(-0.5 * ((t - s1_times)/1.2)**2).sum() + jnp.exp(-0.5 * ((t - s2_time)/1.2)**2)
    I_stim = jnp.where(stim > 0.4, 0.2, 0.0)

    J_in = (h * V * (V - A_CRIT) * (1.0 - V)) / p[0]
    J_out = -(1.0 - h) * (V / p[1])
    dV = J_in + J_out + I_stim

    switch = jax.nn.sigmoid(150.0 * (V - V_GATE))
    dh = ((1.0 - h)/p[2]) * (1.0 - switch) - (h/p[3]) * switch

    V_next = jnp.clip(V + DT * dV, 0.0, 1.0)
    h_next = jnp.clip(h + DT * dh, 0.0, 1.0)
    return (V_next, h_next), V_next

# -----------------------------
# 3. APD calculation
# -----------------------------
def get_restitution_apd(log_params, s2_val):
    def body(carry, i):
        return mms_step_stabilized(carry, i * DT, log_params, s2_val)
    _, v_trace = jax.lax.scan(body, (0.0, 1.0), jnp.arange(MAX_STEPS))
    s2_start_idx = jnp.round(((S1_BEATS - 1) * S1 + s2_val) / DT).astype(jnp.int32)
    s2_window = jax.lax.dynamic_slice(v_trace, (s2_start_idx,), (int(500/DT),))
    return jnp.sum(jax.nn.sigmoid(300.0 * (s2_window - 0.2))) * DT

run_restitution = jax.vmap(get_restitution_apd, in_axes=(None, 0))

# -----------------------------
# 4. Loss function
# -----------------------------
def loss_fn(lp, target_curve, s2_values):
    sim = run_restitution(lp, s2_values)
    return jnp.mean(optax.huber_loss(sim, target_curve, delta=10.0))

# -----------------------------
# 5. Optimization
#    Cosine-decay learning-rate schedule: damps the step size down
#    over the course of training, avoiding overshoot/oscillation
#    near convergence that a fixed, constant learning rate causes.
# -----------------------------
def recover_parameters(target_curve, s2_values, num_iters=NUM_ITERS):
    init_lp = inv_p4(jnp.array([0.1, 1, 100.0, 125.0]))

    lr_schedule = optax.cosine_decay_schedule(
        init_value=0.02, decay_steps=num_iters, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(init_lp)
    lp = init_lp

    for it in range(num_iters):
        g = jax.grad(loss_fn)(lp, target_curve, s2_values)
        updates, opt_state = optimizer.update(g, opt_state)
        lp = optax.apply_updates(lp, updates)

        if it % 10 == 0 or it == num_iters - 1:
            loss_val = float(loss_fn(lp, target_curve, s2_values))
            params = np.array(get_p4(lp))
            print(f"  it={it:3d}  loss={loss_val:.4f}  "
                  f"params={np.round(params, 3)}  "
                  f"lr={float(lr_schedule(it)):.5f}")

    return get_p4(lp)

# -----------------------------
# 6. Run optimization
# -----------------------------
print("\nRunning 0D parameter recovery optimization...")
target_curve = run_restitution(inv_p4(param_true), s2_test)
recovered_params = recover_parameters(target_curve, s2_test)
print(f"\nGround truth params:  {np.array(param_true)}")
print(f"Recovered params:     {np.array(recovered_params)}")

# ── Table 1: APD comparison, GT vs recovered, per S2 point --
#    this is what the loss actually fits, so this is the table that
#    reports fit QUALITY directly. ─────────────────────────────────────
recovered_apd = np.array(run_restitution(inv_p4(recovered_params), s2_test))
gt_apd = np.array(target_curve)
apd_diff = recovered_apd - gt_apd
apd_pct = 100.0 * np.abs(apd_diff) / np.maximum(np.abs(gt_apd), 1e-9)

print(f"\n{'='*60}")
print("APD RESTITUTION FIT: GT vs Recovered")
print(f"{'='*60}")
print(f"  {'S2 (ms)':>10}{'APD GT (ms)':>14}{'APD Rec (ms)':>15}{'Diff (ms)':>12}{'|Diff| (%)':>12}")
for s2_v, gt_v, rec_v, d, p in zip(np.array(s2_test), gt_apd, recovered_apd, apd_diff, apd_pct):
    print(f"  {s2_v:>10.1f}{gt_v:>14.3f}{rec_v:>15.3f}{d:>12.3f}{p:>12.2f}")
print(f"  {'-'*63}")
print(f"  Mean |diff|: {np.mean(np.abs(apd_diff)):.3f} ms  "
      f"({np.mean(apd_pct):.2f}%)   Max |diff|: {np.max(np.abs(apd_diff)):.3f} ms "
      f"({np.max(apd_pct):.2f}%)")

# ── Table 2: parameter recovery, GT vs recovered -- reported
#    honestly: APD-restitution-only fitting does not strongly
#    constrain tau_in/tau_out (which mainly govern AP upstroke
#    shape, not APD duration), so these are NOT expected to align
#    closely, unlike tau_open/tau_close (which directly govern
#    recovery timing, i.e. what APD itself measures). ──────────────────
PNAMES_0D = ["tau_in", "tau_out", "tau_open", "tau_close"]
gt_p = np.array(param_true)
rec_p = np.array(recovered_params)
param_diff_pct = 100.0 * np.abs(rec_p - gt_p) / np.abs(gt_p)

print(f"\n{'='*60}")
print("PARAMETER RECOVERY: GT vs Recovered")
print("(APD-restitution-only fitting does not strongly constrain")
print(" tau_in/tau_out -- see note above the table)")
print(f"{'='*60}")
print(f"  {'Parameter':<12}{'GT':>12}{'Recovered':>14}{'Abs Diff':>12}{'Diff (%)':>12}")
for name, gtv, recv, pct in zip(PNAMES_0D, gt_p, rec_p, param_diff_pct):
    print(f"  {name:<12}{gtv:>12.4f}{recv:>14.4f}{abs(recv-gtv):>12.4f}{pct:>12.2f}")

print("\nDONE")