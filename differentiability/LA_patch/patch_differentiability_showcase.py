# -*- coding: utf-8 -*-
"""
patch_differentiability_showcase.py
==============================================
Loads the extracted patch geometry (patch_geometry.npz) and runs a differentiability
showcase (FD vs jax.grad vs jax.jacfwd) on simulated omnipolar EGM.

NO external package import -- fully self-contained
"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import subprocess
try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = ""

print("=" * 60)
print("3D Patch Differentiability Showcase (real omnipolar EGM)")
print("  FD vs jax.grad vs jax.jacfwd -- geometry from Part 1's")
print("  extraction, NO clinical mesh loaded here")
if gpu_info:
    print(f"  GPU: {gpu_info}")
print(f"  JAX:     {jax.__version__}")
print(f"  Devices: {jax.devices()}")
DEVICE_TAG = jax.default_backend()
print(f"  Device tag (for saved filenames): {DEVICE_TAG}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# [0] PATHS -- EDIT to match where Part 1's output landed
#     (e.g. a Kaggle input dataset, once uploaded there)
# ══════════════════════════════════════════════════════════════════════════
PATCH_GEOMETRY_PATH = "/kaggle/input/datasets/aizaovais/jax-ep-data/jax_ep_data2/jax_ep_data2/data_upload/patch_geometry.npz"  # <- EDIT

_default_out = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "./outputs"
OUT_DIR = os.environ.get("JAX_EP_PATCH_OUTPUT_DIR", _default_out)
os.makedirs(OUT_DIR, exist_ok=True)

GT_SETS = {
    'set1': [0.340, 5.580, 192.0, 120.0, 0.200],
    'set2': [0.260, 3.460, 148.0, 176.0, 0.200],
    'set3': [0.300, 5.000, 120.0, 150.0, 0.200],
}
CASES = ['set1', 'set2', 'set3']
DEFAULT_CASE = 'set3'

V_GATE = 0.13; A_CRIT = 0.13; BETA = 100.; CM = 1.
STIM_AMP = 200.; STIM_DUR = 2.
S1_START = 10.; S2_BCL = 500.
S2_ON = S1_START + S2_BCL; TOTAL_MS = S2_ON + 600.
CLIQUES = np.array([[1, 2, 5, 6], [2, 3, 6, 7], [3, 4, 7, 8],
                     [5, 6, 9, 10], [6, 7, 10, 11], [7, 8, 11, 12],
                     [9, 10, 13, 14], [10, 11, 14, 15], [11, 12, 15, 16]]) - 1
CENTRE_CLIQUE = 4
EDGE_ELEC = {"LEFT": np.array([0, 4, 8, 12]), "RIGHT": np.array([3, 7, 11, 15]),
             "BOTTOM": np.array([0, 1, 2, 3]), "TOP": np.array([12, 13, 14, 15])}
OFFSET_MM = 3.0; STRIP_PAD_MM = 2.0

# ══════════════════════════════════════════════════════════════════════════
# [1] LOAD EXTRACTED PATCH GEOMETRY -- from Part 1, NOT Labelled.pts
# ══════════════════════════════════════════════════════════════════════════
print(f"\nLoading extracted patch geometry: {PATCH_GEOMETRY_PATH}")
_pg = np.load(PATCH_GEOMETRY_PATH, allow_pickle=True)
lv = np.array(_pg["lv"], dtype=np.float64)
le = np.array(_pg["le"], dtype=np.int64)
HD16 = np.array(_pg["HD16"], dtype=np.float64)
fib3 = np.array(_pg["fib3"], dtype=np.float64)
cs_edge = str(_pg["cs_edge"][0])
Np = len(lv)
print(f"  {Np:,} nodes  {len(le):,} elements  pacing edge [{cs_edge}]")

# ══════════════════════════════════════════════════════════════════════════
# [2] FEATURE EXTRACTORS -- unchanged from the original GT-generation code
# ══════════════════════════════════════════════════════════════════════════
def wyatt_ari(egm, i_start, dt, win=4000):
    seg = egm[i_start:i_start + win]
    if len(seg) < 10 or np.ptp(seg) < 1e-9: return np.nan
    d1 = np.gradient(seg, dt); grd = max(1, int(50. / dt))
    i_at = int(np.argmin(d1))
    cap = min(len(d1), i_at + int(400. / dt))
    if i_at + grd >= cap: return np.nan
    i_rt = i_at + grd + int(np.argmax(d1[i_at + grd:cap]))
    return float((i_rt - i_at) * dt)
def get_at(egm, i_start, dt, win=200):
    seg = egm[i_start:i_start + win]
    if len(seg) < 5: return np.nan
    return float(np.argmin(np.gradient(seg, dt)) * dt)
def get_slew(egm, i_start, dt, win=200):
    seg = egm[i_start:i_start + win]
    if len(seg) < 5: return np.nan
    return float(np.max(np.abs(np.gradient(seg, dt))))
def get_p2p(egm, i_start, dt, win=4000):
    seg = egm[i_start:i_start + win]
    if len(seg) < 5: return np.nan
    return float(np.ptp(seg))

# ══════════════════════════════════════════════════════════════════════════
# [3] PATCH BUILDER -- build_patch's EXACT, VERBATIM logic, starting
#     from lv/le/Np/HD16/fib3/cs_edge (loaded above) instead of
#     re-deriving them from Verts/Elems. Everything from here on
#     (FEM assembly, pacing mask, lead-field, _build_w, run_3d,
#     get_egm) is UNCHANGED from the original.
# ══════════════════════════════════════════════════════════════════════════
def build_patch(case):
    DT_SIM = float(0.1)
    N_ION = int(4)
    dth = 0.0
    G_IL_nom = float(GT_SETS[case][4])

    vc = lv * 1e-4
    print(f"  [{case}] {Np:,} nodes  pacing [{cs_edge}]:", end=' ')
    nn = np.zeros((Np, 3)); ar = np.zeros(Np)
    for (i, j, k) in le:
        cr = np.cross(lv[j] - lv[i], lv[k] - lv[i]); a = 0.5 * np.linalg.norm(cr)
        n_ = cr / (np.linalg.norm(cr) + 1e-12)
        for nd in (i, j, k): ar[nd] += a / 3.; nn[nd] += a * n_
    for i in range(Np):
        n = np.linalg.norm(nn[i])
        if n > 1e-12: nn[i] /= n
    m_j = jnp.array(1. / (CM * np.maximum(ar * 1e-8,
                    max(1e-8, np.percentile((ar * 1e-8)[ar > 0], 5)))))
    ec = defaultdict(float)
    for (i, j, k) in le:
        pi, pj, pk = vc[i], vc[j], vc[k]
        for (u, v_, w, a_, b_) in [(i, j, k, pj - pi, pk - pi), (j, i, k, pi - pj, pk - pj),
                                     (k, i, j, pi - pk, pj - pk)]:
            c = np.linalg.norm(np.cross(a_, b_))
            if c < 1e-14: continue
            ec[(min(v_, w), max(v_, w))] += 0.5 * np.dot(a_, b_) / c
    eu = np.array([u for u, v_ in ec.keys()], dtype=np.int32)
    ev = np.array([v_ for u, v_ in ec.keys()], dtype=np.int32)
    ecot = np.array(list(ec.values()), dtype=np.float64)
    ed = vc[ev] - vc[eu]; ed /= np.linalg.norm(ed, axis=1, keepdims=True) + 1e-12
    en = (nn[eu] + nn[ev]) / 2.; en /= np.linalg.norm(en, axis=1, keepdims=True) + 1e-12
    eu_j = jnp.array(eu); ev_j = jnp.array(ev)
    ecot_j = jnp.array(ecot); ed_j = jnp.array(ed); en_j = jnp.array(en)
    fib3_j = jnp.array(np.tile(fib3, (len(eu), 1)) if fib3.ndim == 1 else fib3)
    ep = HD16[EDGE_ELEC[cs_edge]]; ec2 = ep.mean(0)
    c5c = HD16[CLIQUES[CENTRE_CLIQUE]].mean(0)
    wf = (c5c - ec2) / (np.linalg.norm(c5c - ec2) + 1e-12); lc = ec2 - wf * OFFSET_MM * 1000.
    _, _, Ve = np.linalg.svd(ep - ec2, full_matrices=False); ld = Ve[0]
    projs = (ep - ec2) @ ld; hl = max(np.ptp(projs) / 2., 1.) + 2000.
    d = lv - lc; pal = d @ ld; perp = d - np.outer(pal, ld)
    dist = np.linalg.norm(perp, axis=1)
    mk = (dist <= STRIP_PAD_MM * 1000.) & (np.abs(pal) <= hl)
    if mk.sum() < 10: mk = (dist <= STRIP_PAD_MM * 2000.) & (np.abs(pal) <= hl + 2000.)
    mk_j = jnp.array(mk.astype(np.float64))
    print(f"{int(mk.sum())} nodes")
    sig_r = 1  # G_IL_nom/(G_IL_nom+0.2)
    W_np = np.zeros((16, Np))
    for e_ in range(16):
        r_ = np.maximum(np.linalg.norm(vc - HD16[e_] * 1e-4, axis=1), 1e-6)
        W_np[e_] = (1. / r_) * (sig_r / (4. * np.pi))
    W_j = jnp.array(W_np)
    NT = int(TOTAL_MS / DT_SIM); t_ms = np.arange(NT) * DT_SIM
    sv = np.zeros(NT, dtype=bool)
    sv |= (t_ms >= S1_START) & (t_ms < S1_START + STIM_DUR)
    sv |= (t_ms >= S2_ON) & (t_ms < S2_ON + STIM_DUR)
    sv_j = jnp.array(sv)
    i_s1 = int(S1_START / DT_SIM)
    i_s2 = int(S2_ON / DT_SIM)
    win_s1 = int(S2_BCL / DT_SIM)
    win_s2 = min(int(400. / DT_SIM), NT - i_s2)

    def _build_w(g_il):
        g_it = g_il / 4.
        fs = fib3_j - jnp.sum(fib3_j * en_j, axis=1, keepdims=True) * en_j
        fs /= jnp.linalg.norm(fs, axis=1, keepdims=True) + 1e-12
        pp = jnp.cross(en_j, fs); pp /= jnp.linalg.norm(pp, axis=1, keepdims=True) + 1e-12
        fr = fs * jnp.cos(dth) + pp * jnp.sin(dth)
        fr /= jnp.linalg.norm(fr, axis=1, keepdims=True) + 1e-12
        cos2 = jnp.sum(ed_j * fr, axis=1) ** 2
        return jnp.abs(ecot_j) * ((g_il / (BETA * CM)) * cos2 + (g_it / (BETA * CM)) * (1. - cos2))

    @jax.jit
    def run_3d(p5_j):
        g_il = p5_j[4]; w = _build_w(g_il)
        dt_sub = DT_SIM / (2 * N_ION); Iext = (STIM_AMP / CM) * DT_SIM * 1e-3; alpha = DT_SIM / 2.
        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w * x[ev_j] + w * x[eu_j])
            Kx = Kx.at[ev_j].add(-w * x[eu_j] + w * x[ev_j])
            return Kx
        def _ion(V, h):
            sw = jax.nn.sigmoid(150. * (V - V_GATE))
            dh = ((1. - h) / p5_j[2]) * (1. - sw) - (h / p5_j[3]) * sw
            return (jnp.clip(V + dt_sub * ((h * V * (V - A_CRIT) * (1. - V)) / p5_j[0]
                                            - (1. - h) * (V / p5_j[1])), 0., 1.),
                    jnp.clip(h + dt_sub * dh, 0., 1.))
        def _cn(V):
            rhs = V - alpha * m_j * spmv(V)
            Vn, _ = jax.scipy.sparse.linalg.cg(
                lambda x: x + alpha * m_j * spmv(x), rhs, x0=V, tol=1e-6, maxiter=50)
            return jnp.clip(Vn, 0., 1.)
        def scan_fn(c, sv_):
            V, h = c
            for _ in range(N_ION): V, h = _ion(V, h)
            V = jnp.where(sv_, jnp.clip(V + Iext * mk_j, 0., 1.), V); V = _cn(V)
            for _ in range(N_ION): V, h = _ion(V, h)
            return (V, h), W_j @ V
        V0 = jnp.zeros(Np, dtype=jnp.float64); h0 = jnp.ones(Np, dtype=jnp.float64)
        (_, _), phi_T = jax.lax.scan(jax.checkpoint(scan_fn), (V0, h0), sv_j)
        return phi_T

    def get_egm(phi):
        """ORIGINAL, unchanged -- returns np.array. GT generation only."""
        egms = []
        for ci in range(9):
            c_ = CLIQUES[ci]
            egms.append((phi[:, c_[0]] - phi[:, c_[1]]
                         - phi[:, c_[2]] + phi[:, c_[3]]) / 4.)
        return np.array(egms)

    def get_egm_jax(phi):
        """ADDED: JAX-native version, jnp.stack -- stays traceable."""
        egms = []
        for ci in range(9):
            c_ = CLIQUES[ci]
            egms.append((phi[:, c_[0]] - phi[:, c_[1]]
                         - phi[:, c_[2]] + phi[:, c_[3]]) / 4.)
        return jnp.stack(egms)

    return run_3d, get_egm, get_egm_jax, DT_SIM, i_s1, i_s2, win_s1, win_s2, NT


# ══════════════════════════════════════════════════════════════════════════
# [4] GT EGM GENERATION -- unchanged from the original GT-generation code
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GT EGM Generation (all 3 cases)")
print(f"{'=' * 60}")
gt_results = {}
for case in CASES:
    print(f"\n{'-' * 60}")
    print(f"Case: {case}")
    print(f"{'-' * 60}")
    gt = GT_SETS[case]
    run_3d, get_egm, get_egm_jax, DT_SIM, i_s1, i_s2, win_s1, win_s2, NT = build_patch(case)
    print(f"  Running 3D forward at GT params...")
    t0 = time.time()
    phi_gt = np.array(run_3d(jnp.array(gt, dtype=jnp.float64)))
    egms_gt = get_egm(phi_gt)
    egm_c5 = egms_gt[CENTRE_CLIQUE]
    print(f"  Done {time.time() - t0:.1f}s")
    segs_s2_gt = egms_gt[:, i_s2:i_s2 + win_s2]
    segs_s1_gt = egms_gt[:, i_s1:i_s1 + win_s1]
    ari_s1_gt = wyatt_ari(egm_c5, i_s1, DT_SIM, win_s1)
    ari_s2_gt = wyatt_ari(egm_c5, i_s2, DT_SIM, win_s2)
    at_s2_gt = get_at(egm_c5, i_s2, DT_SIM)
    slew_s2_gt = get_slew(egm_c5, i_s2, DT_SIM)
    p2p_s2_gt = get_p2p(egm_c5, i_s2, DT_SIM, win_s2)
    print(f"  S1 ARI={ari_s1_gt:.1f}ms  S2 ARI={ari_s2_gt:.1f}ms")
    print(f"  AT={at_s2_gt:.1f}ms  Slew={slew_s2_gt:.2f}  p2p={p2p_s2_gt:.2f}")
    gt_results[case] = dict(
        run_3d=run_3d, get_egm=get_egm, get_egm_jax=get_egm_jax,
        DT_SIM=DT_SIM, i_s1=i_s1, i_s2=i_s2, win_s1=win_s1, win_s2=win_s2, NT=NT,
        egms_gt=egms_gt, segs_s1_gt=segs_s1_gt, segs_s2_gt=segs_s2_gt,
    )
    out_path = os.path.join(OUT_DIR, f"gt2_egm_{case}.npz")
    np.savez(out_path,
        segs_s2_gt=segs_s2_gt.astype(np.float64), segs_s1_gt=segs_s1_gt.astype(np.float64),
        DT_SIM=np.array([DT_SIM]), i_s1=np.array([i_s1]), i_s2=np.array([i_s2]),
        win_s1=np.array([win_s1]), win_s2=np.array([win_s2]), case=np.array([case]))
    print(f"  Saved: {out_path}")

print(f"\n{'=' * 60}")
print("GT EGM generation complete for all 3 cases")
print(f"{'=' * 60}")


# ══════════════════════════════════════════════════════════════════════════
# [5] DIFFERENTIABILITY SHOWCASE (ADDED) -- on the real omnipolar EGM,
#     for the default case (set3)
# ══════════════════════════════════════════════════════════════════════════
PNAMES = ["tau_in", "tau_out", "tau_open", "tau_close", "G_IL"]
FD_EPS = 1e-3


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
    """Manual, UNBATCHED loop of 5 separate jax.jvp calls -- see
    module docstring for why (confirmed CPU compilation-hang
    avoidance from the cube differentiability work)."""
    n_params = len(p5_64)
    jvp_fn = jax.jit(lambda p, t: jax.jvp(loss_fn, (p,), (t,))[1])
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

    print(f"\n  jax.jacfwd (forward-mode AD, manual jvp loop, time {t_fwd:.1f}s):")
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


print(f"\n{'=' * 60}")
print(f"DIFFERENTIABILITY SHOWCASE -- case '{DEFAULT_CASE}' (default)")
print(f"{'=' * 60}")

r = gt_results[DEFAULT_CASE]
run_3d, get_egm_jax = r['run_3d'], r['get_egm_jax']
i_s2, win_s2 = r['i_s2'], r['win_s2']

p5_nominal = jnp.array(GT_SETS[DEFAULT_CASE], dtype=jnp.float64)


def loss_fn(p5_j64):
    """Loss = mean squared centre-clique EGM within the S2 response
    window -- real omnipolar EGM, same loss structure already
    validated on the 3D cube."""
    phi_T = run_3d(p5_j64)
    egms = get_egm_jax(phi_T)
    c5 = egms[CENTRE_CLIQUE, i_s2:i_s2 + win_s2]
    return jnp.mean(c5 ** 2)


print(f"\n[1] Verifying G_IL differentiability ...")
grad_check = jax.grad(loss_fn)(p5_nominal)
print(f"  d(loss)/d(G_IL) = {float(grad_check[4]):.4e}  "
      f"(genuinely non-zero: {abs(float(grad_check[4])) > 1e-12})")

print(f"\n[2] FD gradient (5 parameters x 2 forward passes each) ...")
fd_grads, t_fd = fd_gradient(loss_fn, np.array(p5_nominal), fd_eps=FD_EPS)
print(f"  Done: {t_fd:.1f}s")

print(f"\n[3] jax.grad (reverse-mode AD) ...")
ad_grads, t_ad, ad_loss_val = ad_gradient(loss_fn, p5_nominal)
print(f"  Done: {t_ad:.1f}s")

print(f"\n[4] jax.jacfwd (forward-mode AD, manual jvp loop) ...")
fwd_grads, t_fwd = jacfwd_gradient(loss_fn, p5_nominal)
print(f"  Done: {t_fwd:.1f}s")

print(f"\n{'=' * 60}")
print("COMPARISON")
print(f"{'=' * 60}")
all_stable, fwd_stable = report_comparison(fd_grads, ad_grads, fwd_grads, t_fd, t_ad, t_fwd)

np.savez(os.path.join(OUT_DIR, f"patch_differentiability_{DEFAULT_CASE}_{DEVICE_TAG}.npz"),
         fd_grads=fd_grads, ad_grads=ad_grads, fwd_grads=fwd_grads,
         t_fd=t_fd, t_ad=t_ad, t_fwd=t_fwd, p5_nominal=np.array(p5_nominal), case=DEFAULT_CASE)
print(f"\nSaved: patch_differentiability_{DEFAULT_CASE}_{DEVICE_TAG}.npz")

# ── Plot the centre-clique EGM at the nominal point ─────────────────────
egm_c5_full = r['egms_gt'][CENTRE_CLIQUE]
t_axis = np.arange(r['NT']) * r['DT_SIM']
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_axis, egm_c5_full, color="#0072B2", lw=1.0)
ax.axvline(S1_START, color="grey", lw=0.6, ls="--", alpha=0.6)
ax.axvline(S2_ON, color="grey", lw=0.6, ls="--", alpha=0.6)
ax.set_title(f"3D Patch -- Centre-clique (C5) omnipolar EGM, case '{DEFAULT_CASE}'")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("EGM (a.u.)")
ax.grid(lw=0.3, alpha=0.3)
plt.tight_layout()
fname = os.path.join(OUT_DIR, f"figure_patch_egm_{DEFAULT_CASE}.png")
fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {fname}")

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print(f"{'=' * 60}")
print(f"  Case: {DEFAULT_CASE}  Device: {DEVICE_TAG}")
print(f"  FD:      {t_fd:.1f}s")
print(f"  jax.grad: {t_ad:.1f}s  (stable: {'YES' if all_stable else 'NO -- ionic stiffness'})")
print(f"  jacfwd:  {t_fwd:.1f}s  (stable: {'YES' if fwd_stable else 'NO'})")
print("\nDONE")
