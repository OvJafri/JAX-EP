# -*- coding: utf-8 -*-
"""
run_differentiability_cpu_gpu.py
=====================================
runs the differentiability
showcase (both the
Raw AP and Bipolar EGM sections, FD vs jax.grad vs jax.jacfwd) on
BOTH CPU and GPU in one execution, then reports a direct summary
comparing the two.

"""


import os
import sys
import subprocess
import time

# ══════════════════════════════════════════════════════════════════════════
WORK_DIR = "/path_to_dir/working" if os.path.isdir("/path_to_dir/working") else "."
_default_out = os.path.join(WORK_DIR, "outputs")
OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", _default_out)
# ══════════════════════════════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
WORKER_PATH = os.path.join(WORK_DIR, "_differentiability_worker_tmp.py")

# ── The full differentiability-showcase script, embedded verbatim as a
#    string -- written out to a
#    temp file at runtime, then run twice via subprocess. ─────────────────
WORKER_SOURCE = r'''
# -*- coding: utf-8 -*-
"""
run_differentiability_showcase.py
================================================

"""
import os
import time
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

WORK_DIR = "/path_to_dir/working" if os.path.isdir("/path_to_dir/working") else "."
_default_out = os.path.join(WORK_DIR, "outputs")
OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", _default_out)

try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = ""

print("=" * 60)
print("3D Cube Differentiability Showcase (raw voltage, V^2 loss)")
print("  FD vs jax.grad vs jax.jacfwd -- Rush-Larsen, matching the")
print("  manuscript's own DT=0.1ms/NT=11,100 differentiability claim")
if gpu_info:
    print(f"  GPU: {gpu_info}")
print(f"  JAX:     {jax.__version__}")
print(f"  Devices: {jax.devices()}")
DEVICE_TAG = jax.default_backend()  # "cpu" or "gpu" -- tags saved
# output files so running this script once per device (e.g. once with
# JAX_PLATFORMS=cpu, once on a GPU-enabled session) preserves
# BOTH results side by side, instead of one overwriting the other --
# needed for the CPU-vs-GPU timing comparison in the companion
# plotting script.
print(f"  Device tag (for saved filenames): {DEVICE_TAG}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# EMBEDDED MODULE CODE
# ══════════════════════════════════════════════════════════════════════════



def build_thin_plate_mesh(lx_mm, ly_mm, thickness_mm, dx_mm):
    """
    Build a closed, watertight, triangulated thin rectangular plate.

    Structure: a regular (nx, ny) grid on the top face (z=thickness)
    and an identical grid on the bottom face (z=0), each triangulated
    by splitting quads (same convention as the 2D flat-plate
    benchmark), connected by 4 triangulated side-wall strips around
    the perimeter.

    Parameters
    ----------
    lx_mm, ly_mm : float
        In-plane dimensions of the plate (mm).
    thickness_mm : float
        Plate thickness (mm), i.e. the z-extent.
    dx_mm : float
        In-plane grid spacing (mm).

    Returns
    -------
    Verts : (N, 3) float64 array
        Node coordinates in mm.
    Elems : (M, 3) int64 array
        Triangle connectivity (node indices into Verts).
    top_grid : (ny, nx) int64 array
        Node-index grid for the top face, for electrode/pacing placement.
    bot_grid : (ny, nx) int64 array
        Node-index grid for the bottom face.
    """
    nx = int(round(lx_mm / dx_mm)) + 1
    ny = int(round(ly_mm / dx_mm)) + 1
    x = np.linspace(0, lx_mm, nx)
    y = np.linspace(0, ly_mm, ny)
    xx, yy = np.meshgrid(x, y)  # shape (ny, nx)

    def grid_face(z_val, node_offset, flip):
        verts = np.column_stack([xx.ravel(), yy.ravel(),
                                  np.full(xx.size, z_val)])
        node_grid = np.arange(xx.size).reshape(ny, nx) + node_offset
        tris = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                bl = node_grid[iy, ix]
                br = node_grid[iy, ix + 1]
                tl = node_grid[iy + 1, ix]
                tr = node_grid[iy + 1, ix + 1]
                if not flip:
                    tris.append([bl, br, tr]); tris.append([bl, tr, tl])
                else:
                    tris.append([bl, tr, br]); tris.append([bl, tl, tr])
        return verts, np.array(tris, dtype=np.int64), node_grid

    n_per_face = nx * ny
    top_verts, top_tris, top_grid = grid_face(thickness_mm, 0, flip=False)
    bot_verts, bot_tris, bot_grid = grid_face(0.0, n_per_face, flip=True)

    def wall_strip(top_row, bot_row):
        tris = []
        for i in range(len(top_row) - 1):
            t0, t1 = top_row[i], top_row[i + 1]
            b0, b1 = bot_row[i], bot_row[i + 1]
            tris.append([t0, t1, b1]); tris.append([t0, b1, b0])
        return np.array(tris, dtype=np.int64)

    walls = [
        wall_strip(top_grid[0, :], bot_grid[0, :]),
        wall_strip(top_grid[ny - 1, :], bot_grid[ny - 1, :]),
        wall_strip(top_grid[:, 0], bot_grid[:, 0]),
        wall_strip(top_grid[:, nx - 1], bot_grid[:, nx - 1]),
    ]

    Verts = np.vstack([top_verts, bot_verts])
    Elems = np.vstack([top_tris, bot_tris] + walls)
    return Verts, Elems, top_grid, bot_grid



def build_edge_pacing_mask(Verts, top_grid, bot_grid, edge="LEFT"):
    """
    Build a pacing mask covering an ENTIRE edge of the plate (both
    top and bottom face nodes along that edge, plus the connecting
    side-wall nodes are automatically included since they reuse the
    same perimeter node indices).

    Parameters
    ----------
    edge : str
        One of "LEFT", "RIGHT", "BOTTOM", "TOP" -- matches the
        EDGE_ELEC convention already used in the patch-builder code
        elsewhere in this repository.

    Returns
    -------
    mask : (N,) float64 array
        1.0 at paced nodes, 0.0 elsewhere.
    paced_node_ids : (K,) int64 array
        The node indices that are paced (for writing a .vtx-style
        file if desired).
    """
    ny, nx = top_grid.shape
    if edge == "LEFT":
        top_ids, bot_ids = top_grid[:, 0], bot_grid[:, 0]
    elif edge == "RIGHT":
        top_ids, bot_ids = top_grid[:, nx - 1], bot_grid[:, nx - 1]
    elif edge == "BOTTOM":
        top_ids, bot_ids = top_grid[0, :], bot_grid[0, :]
    elif edge == "TOP":
        top_ids, bot_ids = top_grid[ny - 1, :], bot_grid[ny - 1, :]
    else:
        raise ValueError(f"Unknown edge '{edge}', expected one of "
                          f"LEFT/RIGHT/BOTTOM/TOP")

    paced_node_ids = np.concatenate([top_ids, bot_ids]).astype(np.int64)
    mask = np.zeros(len(Verts), dtype=np.float64)
    mask[paced_node_ids] = 1.0
    return mask, paced_node_ids






def build_fem_operators(Verts_mm, Elems, CM=1.0):
    """
    Assemble the lumped mass scaling and edge cotangent weights.

    Parameters
    ----------
    Verts_mm : (N, 3) array
        Node coordinates in mm.
    Elems : (M, 3) int array
        Triangle connectivity.
    CM : float
        Membrane capacitance.

    Returns
    -------
    m_inv : (N,) float64 array
        Inverse lumped mass (1 / (CM * nodal_area)).
    eu, ev : (K,) int32 arrays
        Edge node-index pairs.
    ecot : (K,) float64 array
        Cotangent weight per edge.
    ed_x : (K,) float64 array
        x-component of the unit edge direction (for the fibre-
        alignment cos^2 term, computed by the caller).
    """
    Np = len(Verts_mm)
    vc = Verts_mm * 1e-1  # mm -> cm, matches the rest of the codebase

    # Lumped nodal areas
    ar = np.zeros(Np)
    for tri in Elems:
        i, j, k = tri
        cr = np.cross(vc[j] - vc[i], vc[k] - vc[i])
        a = 0.5 * np.linalg.norm(cr)
        for nd in (i, j, k):
            ar[nd] += a / 3.0

    m_inv = 1.0 / (CM * np.maximum(
        ar, max(1e-12, np.percentile(ar[ar > 0], 5))))

    # Edge cotangent weights
    ec = defaultdict(float)
    for tri in Elems:
        i, j, k = tri
        pi, pj, pk = vc[i], vc[j], vc[k]
        for (u, v_, w, a_, b_) in [
            (i, j, k, pj - pi, pk - pi),
            (j, i, k, pi - pj, pk - pj),
            (k, i, j, pi - pk, pj - pk),
        ]:
            c = np.linalg.norm(np.cross(a_, b_))
            if c < 1e-14:
                continue
            ec[(min(v_, w), max(v_, w))] += 0.5 * np.dot(a_, b_) / c

    eu = np.array([e[0] for e in ec], dtype=np.int32)
    ev = np.array([e[1] for e in ec], dtype=np.int32)
    ecot = np.array(list(ec.values()), dtype=np.float64)

    ed = vc[ev] - vc[eu]
    ed_n = ed / (np.linalg.norm(ed, axis=1, keepdims=True) + 1e-12)
    ed_x = ed_n[:, 0]  # fibre direction assumed along global x

    return m_inv, eu, ev, ecot, ed_x


def anisotropic_weights(ecot, ed_x, G_IL, BETA, CM):
    """
    Build the final anisotropic edge conductivity weights, matching
    the 2D benchmark / convergence-study formulation exactly:
    G_IT = G_IL / 4 (4:1 anisotropy ratio).

    Parameters
    ----------
    ecot : (K,) array
        Cotangent weights from build_fem_operators.
    ed_x : (K,) array
        x-component of unit edge direction from build_fem_operators.
    G_IL : float
        Longitudinal conductivity.
    BETA : float
        Surface-to-volume ratio.
    CM : float
        Membrane capacitance.

    Returns
    -------
    w : (K,) array
        Final edge weights for the diffusion operator.
    """
    G_IT = G_IL / 4.0
    cos2 = ed_x ** 2
    return np.abs(ecot) * ((G_IL / (BETA * CM)) * cos2
                            + (G_IT / (BETA * CM)) * (1.0 - cos2))




def _make_w_fn(w_static, ecot, ed_x, BETA, CM):
    """
    Internal helper: returns a function w_fn(p5_j64) -> w (the
    per-edge diffusion weights), either:
      - dynamically, from p5_j64[4] (G_IL), if ecot/ed_x/BETA are
        given (makes G_IL genuinely differentiable), or
      - the fixed, static w_static (backward-compatible fallback)
        if ecot/ed_x/BETA are not given.
    """
    if ecot is not None and ed_x is not None and BETA is not None:
        ecot_j = jnp.array(ecot, dtype=jnp.float64)
        ed_x_j = jnp.array(ed_x, dtype=jnp.float64)
        cos2_j = ed_x_j ** 2

        def w_fn(p5_j64):
            G_IL = p5_j64[4]
            G_IT = G_IL / jnp.float64(4.0)
            return jnp.abs(ecot_j) * (
                (G_IL / (jnp.float64(BETA) * jnp.float64(CM))) * cos2_j
                + (G_IT / (jnp.float64(BETA) * jnp.float64(CM))) * (jnp.float64(1.0) - cos2_j)
            )
        return w_fn, True
    else:
        w_j_static = jnp.array(w_static, dtype=jnp.float64)

        def w_fn(p5_j64):
            return w_j_static
        return w_fn, False





def make_forward_solver_traces(Np, eu, ev, m_inv, w, mask_pace,
                                 node_indices, V_GATE, A_CRIT, DT, N_ION,
                                 STIM_AMP, CM, sv_schedule,
                                 ecot=None, ed_x=None, BETA=None):
    """
    Build a JIT-compiled forward solver that outputs the RAW
    membrane-potential action-potential trace (V, not any lead-field
    projection or activation-time summary) at a small, specified set
    of node indices, over every timestep.

    Uses the EXACT SAME numerical scheme (Rush-Larsen ionic
    integration, Crank-Nicolson diffusion via conjugate gradient) as
    make_forward_solver and make_forward_solver_atmap -- only the
    output differs.

    Parameters
    ----------
    node_indices : (K,) int array
        The specific node indices to record the full V(t) trace at
        (keep K small -- e.g. 2-5 -- since storing per-timestep V at
        many nodes is far more memory-intensive than a lead-field
        projection or an activation-time summary).
    ecot, ed_x, BETA : optional
        Same meaning as in make_forward_solver_atmap / make_forward_solver
        -- if all three are given, G_IL becomes genuinely differentiable
        through the raw voltage trace output.
    (all other parameters match make_forward_solver_atmap)

    Returns
    -------
    run_forward_traces : callable
        run_forward_traces(p5) -> (NT, K) array of V(t) at the
        specified node_indices.
    """
    m_j = jnp.array(m_inv, dtype=jnp.float64)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)
    node_idx_j = jnp.array(node_indices, dtype=jnp.int32)

    w_fn, g_il_differentiable = _make_w_fn(w, ecot, ed_x, BETA, CM)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    @jax.jit
    def run_forward_traces(p5_j64):
        w_j = w_fn(p5_j64)

        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w_j * x[ev_j] + w_j * x[eu_j])
            Kx = Kx.at[ev_j].add(-w_j * x[eu_j] + w_j * x[ev_j])
            return Kx

        def _ion(V, h):
            sw = jax.nn.sigmoid(jnp.float64(150.) * (V - jnp.float64(V_GATE)))
            alpha_h = (jnp.float64(1.) - sw) / p5_j64[2]
            beta_h = sw / p5_j64[3]
            tau_h = jnp.float64(1.) / (alpha_h + beta_h + jnp.float64(1e-10))
            h_inf = alpha_h * tau_h
            h_new = h_inf + (h - h_inf) * jnp.exp(-dt_sub / tau_h)
            h_new = jnp.clip(h_new, jnp.float64(0.), jnp.float64(1.))
            I_in = h * (V * (V - jnp.float64(A_CRIT)) * (jnp.float64(1.) - V)
                        / p5_j64[0])
            I_out = (jnp.float64(1.) - h) * V / p5_j64[1]
            V_new = jnp.clip(V + dt_sub * (I_in - I_out),
                              jnp.float64(0.), jnp.float64(1.))
            return V_new, h_new

        def _cn(V):
            rhs = V - alpha * m_j * spmv(V)
            Vn, _ = jax.scipy.sparse.linalg.cg(
                lambda x: x + alpha * m_j * spmv(x), rhs, x0=V,
                tol=1e-6, maxiter=50)
            return jnp.clip(Vn, jnp.float64(0.), jnp.float64(1.))

        def scan_fn(c, sv_t):
            V, h = c
            for _ in range(N_ION):
                V, h = _ion(V, h)
            V = jnp.where(sv_t,
                           jnp.clip(V + Iext * mask_j,
                                    jnp.float64(0.), jnp.float64(1.)),
                           V)
            V = _cn(V)
            for _ in range(N_ION):
                V, h = _ion(V, h)
            return (V, h), V[node_idx_j]

        V0 = jnp.zeros(Np, dtype=jnp.float64)
        h0 = jnp.ones(Np, dtype=jnp.float64)
        (_, _), V_traces = jax.lax.scan(jax.checkpoint(scan_fn), (V0, h0), sv_j)
        return V_traces

    run_forward_traces.g_il_differentiable = g_il_differentiable
    return run_forward_traces

PNAMES = ["tau_in", "tau_out", "tau_open", "tau_close", "G_IL"]


def fd_gradient(loss_fn, p5_64, fd_eps=1e-3):
    grads = np.zeros(5, dtype=np.float64)
    t0 = time.time()
    for i in range(5):
        p_p = np.array(p5_64); p_p[i] += fd_eps
        p_m = np.array(p5_64); p_m[i] -= fd_eps
        v_p = float(loss_fn(jnp.array(p_p, dtype=jnp.float64)))
        v_m = float(loss_fn(jnp.array(p_m, dtype=jnp.float64)))
        grads[i] = (v_p - v_m) / (2 * fd_eps)
    return grads, time.time() - t0


def ad_gradient(loss_fn, p5_64):
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    _ = grad_fn(p5_64)
    t0 = time.time()
    loss_val, grads = grad_fn(p5_64)
    grads.block_until_ready()
    return np.array(grads), time.time() - t0, float(loss_val)


def jacfwd_gradient(loss_fn, p5_64):

    n_params = len(p5_64)
    jvp_fn = jax.jit(lambda p, t: jax.jvp(loss_fn, (p,), (t,))[1])
    # Warm-up / compile with the first tangent direction
    tangent0 = jnp.zeros(n_params, dtype=jnp.float64).at[0].set(1.0)
    _ = jvp_fn(p5_64, tangent0)
    t0 = time.time()
    grads = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        tangent = jnp.zeros(n_params, dtype=jnp.float64).at[i].set(1.0)
        grads[i] = float(jvp_fn(p5_64, tangent))
    return grads, time.time() - t0


def report_comparison(fd_grads, ad_grads, fwd_grads, t_fd, t_ad, t_fwd,
                       explode_threshold=10.0, stable_tol=0.15):
    print(f"\n  FD gradients (time {t_fd:.1f}s):")
    for i, pn in enumerate(PNAMES):
        print(f"    {pn:<12}  FD={fd_grads[i]:>12.4e}")

    print(f"\n  jax.grad (reverse-mode AD, time {t_ad:.1f}s):")
    print(f"  {'Param':<12}{'AD grad':>14}{'FD grad':>14}{'ratio':>10}{'stable':>10}")
    all_stable = True
    for i, pn in enumerate(PNAMES):
        ratio = ad_grads[i] / fd_grads[i] if abs(fd_grads[i]) > 1e-10 else float("nan")
        stable = abs(ratio) <= explode_threshold if np.isfinite(ratio) else False
        if not stable: all_stable = False
        print(f"  {pn:<12}{ad_grads[i]:>14.4e}{fd_grads[i]:>14.4e}"
              f"{ratio:>10.4f}{'OK' if stable else 'EXPLODE':>10}")
    print(f"  AD gradients stable: {'YES' if all_stable else 'NO -- ionic stiffness blow-up'}")

    print(f"\n  jax.jacfwd (forward-mode AD, time {t_fwd:.1f}s):")
    print(f"  {'Param':<12}{'jacfwd':>14}{'FD':>14}{'ratio':>10}{'stable':>10}")
    fwd_stable = True
    small_thresh = 1e-10
    for i, pn in enumerate(PNAMES):
        fd_small = abs(fd_grads[i]) <= small_thresh
        fwd_small = abs(fwd_grads[i]) <= small_thresh
        if fd_small and fwd_small:
            ratio = float("nan"); ok = True; status = "OK (both~0)"
        else:
            ratio = fwd_grads[i] / fd_grads[i] if not fd_small else float("nan")
            ok = np.isfinite(ratio) and abs(ratio - 1.0) < stable_tol
            status = "OK" if ok else "FAIL"
        if not ok: fwd_stable = False
        print(f"  {pn:<12}{fwd_grads[i]:>14.4e}{fd_grads[i]:>14.4e}"
              f"{ratio:>10.4f}{status:>12}")
    print(f"  jacfwd stable: {'YES -- fully differentiable' if fwd_stable else 'NO'}")
    return all_stable, fwd_stable


# ══════════════════════════════════════════════════════════════════════════
# MAIN SCRIPT
# ══════════════════════════════════════════════════════════════════════════
LX_MM, LY_MM, THICKNESS_MM, DX_MM = 19.0, 19.0, 1.5, 0.2
PACE_EDGE = "LEFT"
V_GATE, A_CRIT, BETA, CM = 0.13, 0.13, 140.0, 1.0
G_IL_NOMINAL = 0.350
STIM_AMP, STIM_DUR = 200.0, 2.0
N_ION, DT = 4, 0.1
N_T = 11100
TOTAL_MS = N_T * DT
FD_EPS = 1e-3

print(f"\n  Geometry: {LX_MM}x{LY_MM}x{THICKNESS_MM}mm, dx={DX_MM}mm")
print(f"  Protocol: DT={DT}ms  N_T={N_T}  TOTAL={TOTAL_MS}ms  (matches manuscript)")

print("\n[1] Building thin-plate mesh + FEM ...")
t0 = time.time()
Verts, Elems, top_grid, bot_grid = build_thin_plate_mesh(
    LX_MM, LY_MM, THICKNESS_MM, DX_MM)
mask_pace, paced_ids = build_edge_pacing_mask(
    Verts, top_grid, bot_grid, edge=PACE_EDGE)
m_inv, eu, ev, ecot, ed_x = build_fem_operators(Verts, Elems, CM=CM)
w = anisotropic_weights(ecot, ed_x, G_IL=G_IL_NOMINAL, BETA=BETA, CM=CM)
print(f"  {len(Verts):,} nodes  {len(eu):,} edges  ({time.time()-t0:.1f}s)")

# Two trace nodes: one near the pacing edge, one farther across the
# plate -- gives G_IL's conduction-speed effect room to show up
x_coords = Verts[:, 0]
node_near = np.argmin(np.abs(x_coords - LX_MM * 0.35))
node_far = np.argmin(np.abs(x_coords - LX_MM * 0.65))
node_indices = np.array([node_near, node_far])
print(f"  Trace nodes: x={Verts[node_near,0]:.2f}mm, x={Verts[node_far,0]:.2f}mm")

t_ms = np.arange(N_T) * DT
sv = (t_ms >= 10.0) & (t_ms < 12.0)

# G_IL genuinely differentiable: ecot/ed_x/BETA passed so the
# diffusion weights are computed inside the JIT-traced function from
# p5[4], not a fixed external constant
run_traces = make_forward_solver_traces(
    Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, w=w, mask_pace=mask_pace,
    node_indices=node_indices, V_GATE=V_GATE, A_CRIT=A_CRIT, DT=DT,
    N_ION=N_ION, STIM_AMP=STIM_AMP, CM=CM, sv_schedule=sv,
    ecot=ecot, ed_x=ed_x, BETA=BETA)
print(f"\n  G_IL genuinely differentiable: {run_traces.g_il_differentiable}")

p5_nominal = jnp.array(
    [0.300, 5.000, 120.0, 150.0, G_IL_NOMINAL], dtype=jnp.float64)

def loss_fn(p5_j64):
    """Loss = mean(V^2) across both trace nodes, full simulation --
    matches the manuscript's own differentiability-demonstration
    loss shape exactly."""
    V_traces = run_traces(p5_j64)
    return jnp.mean(V_traces ** 2)

print("\n[2] Running forward pass (warm-up) ...")
t0 = time.time()
V_traces_nom = run_traces(p5_nominal)
V_traces_nom.block_until_ready()
t_fwd_pass = time.time() - t0
print(f"  Done: {t_fwd_pass:.1f}s")

np.savez(os.path.join(OUT_DIR, "cube_traces_v2.npz"),
         V_traces=np.array(V_traces_nom), DT=DT, N_T=N_T,
         node_indices=node_indices)

print(f"\n[3] FD gradient (5 parameters x 2 forward passes each) ...")
fd_grads, t_fd = fd_gradient(loss_fn, np.array(p5_nominal), fd_eps=FD_EPS)
print(f"  Done: {t_fd:.1f}s")

print(f"\n[4] jax.grad (reverse-mode AD) ...")
ad_grads, t_ad, ad_loss_val = ad_gradient(loss_fn, p5_nominal)
print(f"  Done: {t_ad:.1f}s")

print(f"\n[5] jax.jacfwd (forward-mode AD, 5 exact tangent passes) ...")
fwd_grads, t_fwd = jacfwd_gradient(loss_fn, p5_nominal)
print(f"  Done: {t_fwd:.1f}s")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
all_stable, fwd_stable = report_comparison(fd_grads, ad_grads, fwd_grads, t_fd, t_ad, t_fwd)

np.savez(os.path.join(OUT_DIR, f"cube_differentiability_v2_{DEVICE_TAG}.npz"),
         fd_grads=fd_grads, ad_grads=ad_grads, fwd_grads=fwd_grads,
         t_fd=t_fd, t_ad=t_ad, t_fwd=t_fwd, p5_nominal=np.array(p5_nominal))
print(f"\nSaved: cube_differentiability_v2_{DEVICE_TAG}.npz")

# ── Plot the two AP traces ───────────────────────────────────────────────
print(f"\n[6] Plotting AP traces ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
t_axis = np.arange(N_T) * DT
V_np = np.array(V_traces_nom)
for i, ax in enumerate(axes):
    ax.plot(t_axis, V_np[:, i], color="#0072B2", lw=1.0)
    ax.set_title(f"Node at x={Verts[node_indices[i],0]:.1f}mm")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (normalised)")
    ax.grid(lw=0.3, alpha=0.3)
fig.suptitle("3D Cube -- Raw AP traces at 2 nodes", fontweight="bold")
plt.tight_layout()
fname = os.path.join(OUT_DIR, "figure_cube_ap_traces.png")
fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════
# ADDITIONAL DIFF. SHOWCASE: bipolar EGM, in addition to the raw-AP/V^2 showcase
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ADDITIONAL SHOWCASE: Bipolar EGM (2 electrodes, 1mm wide, 3mm gap)")
print(f"{'='*60}")

mid_x = LX_MM / 2.0
ELEC_WIDTH_MM = 1.0
GAP_MM = 3.0
_total_span = 2 * ELEC_WIDTH_MM + GAP_MM
_x0 = mid_x - _total_span / 2.0
E1_X_RANGE = (_x0, _x0 + ELEC_WIDTH_MM)
E2_X_RANGE = (_x0 + ELEC_WIDTH_MM + GAP_MM, _x0 + ELEC_WIDTH_MM + GAP_MM + ELEC_WIDTH_MM)
mid_y = LY_MM / 2.0
Y_HALF_WIDTH_MM = 1.0  # electrode also spans +-1mm in y, around the centreline

print(f"  Electrode 1: x=[{E1_X_RANGE[0]:.2f},{E1_X_RANGE[1]:.2f}]mm  "
      f"y=[{mid_y-Y_HALF_WIDTH_MM:.2f},{mid_y+Y_HALF_WIDTH_MM:.2f}]mm")
print(f"  Electrode 2: x=[{E2_X_RANGE[0]:.2f},{E2_X_RANGE[1]:.2f}]mm  "
      f"y=[{mid_y-Y_HALF_WIDTH_MM:.2f},{mid_y+Y_HALF_WIDTH_MM:.2f}]mm")
print(f"  Centre-to-centre distance: "
      f"{((E2_X_RANGE[0]+E2_X_RANGE[1])/2)-((E1_X_RANGE[0]+E1_X_RANGE[1])/2):.2f}mm")

# Restrict to top-face nodes only (electrodes sit on the surface)
n_per_face = len(Verts) // 2
top_ids = np.arange(n_per_face)
x_top, y_top = Verts[top_ids, 0], Verts[top_ids, 1]

e1_sel = (x_top >= E1_X_RANGE[0]) & (x_top <= E1_X_RANGE[1]) & \
         (np.abs(y_top - mid_y) <= Y_HALF_WIDTH_MM)
e2_sel = (x_top >= E2_X_RANGE[0]) & (x_top <= E2_X_RANGE[1]) & \
         (np.abs(y_top - mid_y) <= Y_HALF_WIDTH_MM)
e1_nodes = top_ids[e1_sel]
e2_nodes = top_ids[e2_sel]
n_e1 = len(e1_nodes)
print(f"  Electrode 1: {n_e1} nodes    Electrode 2: {len(e2_nodes)} nodes")
assert n_e1 > 0 and len(e2_nodes) > 0, "Electrode selection found no nodes -- check geometry/mesh resolution"

bipolar_node_indices = np.concatenate([e1_nodes, e2_nodes])

# RECORDING HEIGHT: each electrode records from a virtual point
# RECORDING_HEIGHT_MM above the surface, directly over its own node
# set's own centroid -- weighted by 1/r, LOCALIZED to that electrode's
# own (small) node set only, NOT the whole patch (which would
# reintroduce the earlier, confirmed-problematic whole-mesh
# lead-field morphology issue). This is a genuine, meaningful,
# NORMALIZED weighted AVERAGE (weights sum to 1), not an unweighted
# lead-field sum -- nodes directly below the virtual point get more
# weight than nodes at the electrode's own edges.
RECORDING_HEIGHT_MM = 0.2
z_surface = Verts[e1_nodes[0], 2]
e1_centroid = Verts[e1_nodes].mean(axis=0)
e2_centroid = Verts[e2_nodes].mean(axis=0)
e1_virtual_pt = np.array([e1_centroid[0], e1_centroid[1], z_surface + RECORDING_HEIGHT_MM])
e2_virtual_pt = np.array([e2_centroid[0], e2_centroid[1], z_surface + RECORDING_HEIGHT_MM])

def _local_weights(nodes, virtual_pt):
    r = np.maximum(np.linalg.norm(Verts[nodes] - virtual_pt, axis=1), 1e-6)
    w_ = 1.0 / r
    return w_ / w_.sum()

e1_weights = jnp.array(_local_weights(e1_nodes, e1_virtual_pt), dtype=jnp.float64)
e2_weights = jnp.array(_local_weights(e2_nodes, e2_virtual_pt), dtype=jnp.float64)
print(f"  Recording height: {RECORDING_HEIGHT_MM}mm above the surface")

# Same solver factory, same G_IL-differentiability mechanism, just a
# different set of traced nodes
run_bipolar_traces = make_forward_solver_traces(
    Np=len(Verts), eu=eu, ev=ev, m_inv=m_inv, w=w, mask_pace=mask_pace,
    node_indices=bipolar_node_indices, V_GATE=V_GATE, A_CRIT=A_CRIT, DT=DT,
    N_ION=N_ION, STIM_AMP=STIM_AMP, CM=CM, sv_schedule=sv,
    ecot=ecot, ed_x=ed_x, BETA=BETA)
print(f"\n  G_IL genuinely differentiable (bipolar): "
      f"{run_bipolar_traces.g_il_differentiable}")

def compute_bipolar(V_all):
    """V_all: (NT, n_e1+n_e2) raw per-node voltage. Returns the
    bipolar EGM (NT,): recording-height-weighted electrode-2 average
    minus recording-height-weighted electrode-1 average -- INVERTED
    polarity relative to the original (v1-v2) convention."""
    v1 = V_all[:, :n_e1] @ e1_weights
    v2 = V_all[:, n_e1:] @ e2_weights
    return v2 - v1  # inverted polarity: v2 - v1, not v1 - v2

def bipolar_loss_fn(p5_j64):
    V_all = run_bipolar_traces(p5_j64)
    bipolar = compute_bipolar(V_all)
    return jnp.mean(bipolar ** 2)

print("\n  [1] Running forward pass (warm-up) ...")
t0 = time.time()
V_all_nom = run_bipolar_traces(p5_nominal)
V_all_nom.block_until_ready()
t_fwd_pass_bipolar = time.time() - t0
print(f"    Done: {t_fwd_pass_bipolar:.1f}s")

bipolar_nom = np.array(compute_bipolar(V_all_nom))
np.savez(os.path.join(OUT_DIR, "cube_bipolar_v2.npz"),
         bipolar=bipolar_nom, DT=DT, N_T=N_T,
         e1_nodes=e1_nodes, e2_nodes=e2_nodes)

print("\n  [2] FD gradient (5 parameters x 2 forward passes each) ...")
fd_grads_bp, t_fd_bp = fd_gradient(bipolar_loss_fn, np.array(p5_nominal), fd_eps=FD_EPS)
print(f"    Done: {t_fd_bp:.1f}s")

print("\n  [3] jax.grad (reverse-mode AD) ...")
ad_grads_bp, t_ad_bp, ad_loss_val_bp = ad_gradient(bipolar_loss_fn, p5_nominal)
print(f"    Done: {t_ad_bp:.1f}s")

print("\n  [4] jax.jacfwd (forward-mode AD, 5 exact tangent passes) ...")
fwd_grads_bp, t_fwd_bp = jacfwd_gradient(bipolar_loss_fn, p5_nominal)
print(f"    Done: {t_fwd_bp:.1f}s")

print(f"\n{'-'*60}")
print("BIPOLAR EGM COMPARISON")
print(f"{'-'*60}")
all_stable_bp, fwd_stable_bp = report_comparison(
    fd_grads_bp, ad_grads_bp, fwd_grads_bp, t_fd_bp, t_ad_bp, t_fwd_bp)

np.savez(os.path.join(OUT_DIR, f"cube_differentiability_bipolar_{DEVICE_TAG}.npz"),
         fd_grads=fd_grads_bp, ad_grads=ad_grads_bp, fwd_grads=fwd_grads_bp,
         t_fd=t_fd_bp, t_ad=t_ad_bp, t_fwd=t_fwd_bp,
         p5_nominal=np.array(p5_nominal))
print(f"\n  Saved: cube_differentiability_bipolar_{DEVICE_TAG}.npz")

# ── Plot the bipolar EGM ─────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
ax.plot(t_axis, bipolar_nom, color="#D55E00", lw=1.0)
ax.set_title("3D Cube -- Bipolar EGM (2x 1mm electrodes, 3mm gap)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Bipolar EGM (V2 - V1, inverted, normalised)")
ax.grid(lw=0.3, alpha=0.3)
plt.tight_layout()
fname_bp = os.path.join(OUT_DIR, "figure_cube_bipolar_egm.png")
fig.savefig(fname_bp, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {fname_bp}")

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Mesh: {len(Verts):,} nodes  {len(Elems):,} elements")
print(f"\n  -- Raw AP (V^2) showcase --")
print(f"  G_IL differentiable: {run_traces.g_il_differentiable}")
print(f"  Forward pass: {t_fwd_pass:.1f}s  (device: {jax.devices()[0]})")
print(f"  FD:      {t_fd:.1f}s")
print(f"  jax.grad: {t_ad:.1f}s  (stable: {'YES' if all_stable else 'NO -- ionic stiffness'})")
print(f"  jacfwd:  {t_fwd:.1f}s  (stable: {'YES' if fwd_stable else 'NO'})")
print(f"\n  -- Bipolar EGM showcase --")
print(f"  G_IL differentiable: {run_bipolar_traces.g_il_differentiable}")
print(f"  Forward pass: {t_fwd_pass_bipolar:.1f}s  (device: {jax.devices()[0]})")
print(f"  FD:      {t_fd_bp:.1f}s")
print(f"  jax.grad: {t_ad_bp:.1f}s  (stable: {'YES' if all_stable_bp else 'NO -- ionic stiffness'})")
print(f"  jacfwd:  {t_fwd_bp:.1f}s  (stable: {'YES' if fwd_stable_bp else 'NO'})")
print("\nDONE")
'''

with open(WORKER_PATH, "w") as f:
    f.write(WORKER_SOURCE)
print(f"Wrote worker to: {WORKER_PATH}")


def run_worker(force_cpu):
    label = "CPU" if force_cpu else "GPU"
    print(f"\n{'='*60}")
    print(f"Running {label} pass (subprocess) ...")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["JAX_EP_CUBE_OUTPUT_DIR"] = OUT_DIR
    if force_cpu:
        env["JAX_PLATFORMS"] = "cpu"
    elif "JAX_PLATFORMS" in env:
        del env["JAX_PLATFORMS"]

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, WORKER_PATH], env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    if result.returncode != 0 or "DONE" not in result.stdout:
        raise RuntimeError(
            f"{label} subprocess failed (exit code {result.returncode}) "
            f"-- see output above.")

    print(f"  [{label} subprocess wall-clock: {elapsed:.1f}s total]")
    return elapsed


t_cpu_wall = run_worker(force_cpu=True)
t_gpu_wall = run_worker(force_cpu=False)

# ── Load both devices' results directly and print a compact summary
#    (the full, detailed comparison figure is built separately by
#    plot_gradient_verification_cube.py, which already knows how to
#    find and use both _cpu.npz and _gpu.npz files automatically) ────────
import numpy as np

print(f"\n{'='*60}")
print("CPU vs GPU SUMMARY")
print(f"{'='*60}")

for base_name, label in [("cube_differentiability_v2", "Raw AP (V^2 loss)"),
                          ("cube_differentiability_bipolar", "Bipolar EGM")]:
    cpu_path = os.path.join(OUT_DIR, f"{base_name}_cpu.npz")
    gpu_path = os.path.join(OUT_DIR, f"{base_name}_gpu.npz")
    if not (os.path.isfile(cpu_path) and os.path.isfile(gpu_path)):
        print(f"\n  {label}: missing one or both device result files "
              f"-- skipping summary for this showcase.")
        continue
    c = np.load(cpu_path)
    g = np.load(gpu_path)
    cpu_total = float(c["t_fd"]) + float(c["t_ad"]) + float(c["t_fwd"])
    gpu_total = float(g["t_fd"]) + float(g["t_ad"]) + float(g["t_fwd"])
    speedup = cpu_total / gpu_total if gpu_total > 0 else float("nan")
    print(f"\n  {label}:")
    print(f"    CPU total (FD+jax.grad+jacfwd): {cpu_total:.1f}s")
    print(f"    GPU total (FD+jax.grad+jacfwd): {gpu_total:.1f}s")
    print(f"    Speedup: {speedup:.1f}x")

print(f"\n{'='*60}")
print(f"Both devices' result files are now saved in: {OUT_DIR}")
print(f"{'='*60}")
print("\nDONE")