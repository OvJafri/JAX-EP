# -*- coding: utf-8 -*-
"""
differentiability.py
=====================
FD / jax.grad / jax.jacfwd gradient comparison on the 3D cube,
reproducing the same comparison structure as the patch-level
demonstration this cube replaces.

Loss function: mean squared error of the centre-clique OEGM within
the S2 response window (matching _loss_patch in the original code).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from .solver import compute_oegm

PNAMES = ['tau_in', 'tau_out', 'tau_open', 'tau_close', 'G_IL']


def make_atmap_loss_fn(run_forward_atmap, node_indices):
    """
    Build a scalar loss function of p5, based on the VALIDATED,
    dV/dt-weighted soft activation-time map (at_soft) from
    make_forward_solver_atmap -- NOT the lead-field/OEGM path (which
    still has an open, unresolved mesh-density-dependence question;
    see mesh.py/solver.py notes).

    Loss = mean(at_soft) over the given node_indices -- directly
    analogous to the main forward_model's own "dAT/dG_IL full-LA
    sensitivity" section, which differentiates activation time
    itself with respect to parameters (not an EGM-derived quantity).

    Parameters
    ----------
    run_forward_atmap : callable
        From make_forward_solver_atmap.
    node_indices : array-like of int
        Which nodes' at_soft values to average into the loss (keep
        this small and representative, e.g. a few nodes spread across
        the plate, not the full mesh -- FD needs 5 full forward
        passes per parameter regardless of loss complexity, so this
        doesn't change FD's cost, but keeps the loss itself simple
        and directly interpretable).
    """
    node_idx_j = jnp.array(node_indices, dtype=jnp.int32)

    def loss_fn(p5_j64):
        at_hard, at_soft, activated = run_forward_atmap(p5_j64)
        return jnp.mean(at_soft[node_idx_j])
    return loss_fn


def make_loss_fn(run_forward, cliques, centre_clique_idx, i_s2, win_s2):
    """
    Build a scalar loss function of p5 (the 5 recovered parameters),
    matching the patch's _loss_patch: mean squared value of the
    centre-clique OEGM within the S2 window.
    """
    def loss_fn(p5_j64):
        phi_T = run_forward(p5_j64)
        oegms = compute_oegm(phi_T, cliques)
        c5 = oegms[centre_clique_idx, i_s2:i_s2 + win_s2]
        return jnp.mean(c5 ** 2)
    return loss_fn


def fd_gradient(loss_fn, p5_64, fd_eps=1e-3):
    """Central finite-difference gradient, one component at a time."""
    grads = np.zeros(5, dtype=np.float64)
    t0 = time.time()
    for i in range(5):
        p_p = np.array(p5_64); p_p[i] += fd_eps
        p_m = np.array(p5_64); p_m[i] -= fd_eps
        v_p = float(loss_fn(jnp.array(p_p, dtype=jnp.float64)))
        v_m = float(loss_fn(jnp.array(p_m, dtype=jnp.float64)))
        grads[i] = (v_p - v_m) / (2 * fd_eps)
    t_fd = time.time() - t0
    return grads, t_fd


def ad_gradient(loss_fn, p5_64):
    """jax.grad (reverse-mode AD) gradient -- expected to be UNSTABLE
    over the full, stiff ionic time-chain for full-resolution
    problems, matching the manuscript's own finding."""
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    _ = grad_fn(p5_64)  # warm up
    t0 = time.time()
    loss_val, grads = grad_fn(p5_64)
    grads.block_until_ready()
    t_ad = time.time() - t0
    return np.array(grads), t_ad, float(loss_val)


def jacfwd_gradient(loss_fn, p5_64):
    """jax.jacfwd (forward-mode AD) gradient -- expected to be
    stable, exact, and to match FD closely."""
    jacfwd_fn = jax.jit(jax.jacfwd(loss_fn))
    _ = jacfwd_fn(p5_64)  # warm up
    t0 = time.time()
    grads = np.array(jacfwd_fn(p5_64))
    t_fwd = time.time() - t0
    return grads, t_fwd


def report_comparison(fd_grads, ad_grads, fwd_grads, t_fd, t_ad, t_fwd,
                       explode_threshold=10.0, stable_tol=0.15):
    """Print the same 3-way comparison table format used in the
    original patch code."""
    print(f"\n  FD gradients (time {t_fd:.1f}s):")
    for i, pn in enumerate(PNAMES):
        print(f"    {pn:<12}  FD={fd_grads[i]:>12.4e}")

    print(f"\n  jax.grad (reverse-mode AD, time {t_ad:.1f}s):")
    print(f"  {'Param':<12}{'AD grad':>14}{'FD grad':>14}{'ratio':>10}{'stable':>10}")
    all_stable = True
    for i, pn in enumerate(PNAMES):
        ratio = ad_grads[i] / fd_grads[i] if abs(fd_grads[i]) > 1e-10 else float('nan')
        stable = abs(ratio) <= explode_threshold if np.isfinite(ratio) else False
        if not stable:
            all_stable = False
        print(f"  {pn:<12}{ad_grads[i]:>14.4e}{fd_grads[i]:>14.4e}"
              f"{ratio:>10.4f}{'OK' if stable else 'EXPLODE':>10}")
    print(f"  AD gradients stable: {'YES' if all_stable else 'NO -- ionic stiffness blow-up'}")

    print(f"\n  jax.jacfwd (forward-mode AD, time {t_fwd:.1f}s):")
    print(f"  {'Param':<12}{'jacfwd':>14}{'FD':>14}{'ratio':>10}{'stable':>10}")
    fwd_stable = True
    for i, pn in enumerate(PNAMES):
        ratio = fwd_grads[i] / fd_grads[i] if abs(fd_grads[i]) > 1e-10 else float('nan')
        ok = np.isfinite(ratio) and abs(ratio - 1.0) < stable_tol
        if not ok:
            fwd_stable = False
        print(f"  {pn:<12}{fwd_grads[i]:>14.4e}{fd_grads[i]:>14.4e}"
              f"{ratio:>10.4f}{'OK' if ok else 'FAIL':>10}")
    print(f"  jacfwd stable: {'YES -- fully differentiable' if fwd_stable else 'NO'}")

    return all_stable, fwd_stable
