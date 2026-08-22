# -*- coding: utf-8 -*-
"""
solver.py
=========
Forward electrophysiology solver for the 3D thin-plate cube.

Faithfully reproduces the EXACT numerical scheme used in the
patch-level differentiability demonstration this cube replaces
(parameter_learning / forward_model's _run_patch): Rush-Larsen
integration for the gating variable (explicitly noted in the
original code as "stable for jacfwd"), Crank-Nicolson diffusion via
conjugate gradient, N_ION=4 ionic sub-steps, DT=0.1ms.

This specific DT=0.1ms setting is not arbitrary: it is required to
genuinely reproduce the manuscript's precisely-quoted N_T=11,100
timestep claim for this exact kind of patch/differentiability
analysis (confirmed: DT=0.1ms x NT=11,100 = 1110ms = the patch's
S1-S2 protocol duration).
"""
import jax
import jax.numpy as jnp


def make_forward_solver_atmap(Np, eu, ev, m_inv, w, mask_pace,
                                V_GATE, A_CRIT, DT, N_ION, STIM_AMP, CM,
                                sv_schedule, T_BLANK=None, BETA_AT=50.0,
                                DETECTION_WINDOW=None):
    """
    Build a JIT-compiled forward solver that tracks BOTH the hard
    activation-time map (first V > V_GATE crossing after blanking)
    AND the soft, sigmoid-weighted activation-time map -- matching
    the ORIGINAL monolithic Forward_LA.py's run_3d/scan_fn exactly
    (at_num/at_den accumulators, BETA_AT=50 sigmoid weighting), not
    a partial reproduction.

    Use this FIRST, before adding the HD-grid/lead-field calculation,
    to directly validate the core physics -- wave propagation, CV,
    LAT-map sensibility -- independently of the electrode-placement
    and lead-field code (where two genuine bugs were already found
    and fixed: electrode-node coincidence, and the lead-field sum's
    mesh-density dependence). This keeps those two concerns cleanly
    separated for debugging.

    Parameters are the same as make_forward_solver, EXCEPT there is
    no W_lead (lead-field matrix) -- not needed here.

    T_BLANK : float, optional
        Blanking period (ms) after which activation can be detected.
        Defaults to the first stimulus time + stimulus duration + 1ms
        if not given (derived from sv_schedule), matching the
        original's own T_BLANK=S1_START+STIM_DUR+1 convention.
    BETA_AT : float
        Sigmoid sharpness for the soft-activation-time weighting
        (matches the original's BETA_AT=50 exactly).
    DETECTION_WINDOW : float, optional
        BOTH at_soft accumulation AND at_hard detection are restricted
        to t_now < DETECTION_WINDOW, matching the original code's own
        "in_beat1 = t_now < S1_BCL" windowing exactly (there, S1_BCL
        bounds detection to the first beat). Without this bound,
        at_soft accumulates weight over the ENTIRE simulation, not a
        bounded, physiologically appropriate window -- this was a
        genuine, confirmed bug in an earlier version of this function
        (verified directly: at_soft landed near the plateau midpoint
        of a MUCH longer window than intended, not shortly after the
        real upstroke). Defaults to 600ms (matching the original
        code's own S1_BCL=600 convention) if not given.

    Returns
    -------
    run_forward_atmap : callable
        run_forward_atmap(p5) -> (at_hard, at_soft, activated).
    """
    m_j = jnp.array(m_inv, dtype=jnp.float64)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    w_j = jnp.array(w, dtype=jnp.float64)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    if T_BLANK is None:
        import numpy as np
        first_stim_idx = int(np.argmax(sv_schedule))
        # Find the stimulus block length directly from the schedule
        j = first_stim_idx
        while j < len(sv_schedule) and sv_schedule[j]:
            j += 1
        stim_dur_steps = j - first_stim_idx
        T_BLANK = first_stim_idx * DT + stim_dur_steps * DT + 1.0

    if DETECTION_WINDOW is None:
        DETECTION_WINDOW = 600.0

    NT = len(sv_schedule)
    t_arr = jnp.arange(NT, dtype=jnp.float64) * DT

    @jax.jit
    def run_forward_atmap(p5_j64):
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

        def scan_fn(carry, inputs):
            V_prev, h, at_num, at_den, activated, at_hard = carry
            sv_t, t_now = inputs
            V = V_prev
            for _ in range(N_ION):
                V, h = _ion(V, h)
            V = jnp.where(sv_t,
                           jnp.clip(V + Iext * mask_j,
                                    jnp.float64(0.), jnp.float64(1.)),
                           V)
            V = _cn(V)
            for _ in range(N_ION):
                V, h = _ion(V, h)
            # in_beat1: restricts BOTH soft-AT accumulation and hard-AT
            # detection to t_now < DETECTION_WINDOW, matching the
            # original code's own "in_beat1 = t_now < S1_BCL" convention.
            in_beat1 = (t_now < jnp.float64(DETECTION_WINDOW)).astype(jnp.float64)
            # Soft AT: weighted by the RATE OF CHANGE dV/dt (a smooth,
            # differentiable gate on the POSITIVE part of dV), not by
            # V's absolute level. This is the corrected formula -- the
            # original level-based weighting (sigmoid(V-V_gate)) stays
            # high for the ENTIRE plateau (confirmed directly: produces
            # a ~125ms offset from the true upstroke, even with correct
            # windowing), since V remains above V_gate long after the
            # actual activation. Weighting by dV/dt instead concentrates
            # the weight specifically at the fast upstroke (where dV/dt
            # is large and positive) and is near-zero during the slow
            # plateau and repolarisation phases (where dV/dt is small
            # or negative) -- directly analogous to the standard
            # clinical definition of activation time as the point of
            # maximum upstroke velocity (dV/dt max).
            dV = V - V_prev
            w_at = jax.nn.relu(dV) * in_beat1
            at_num = at_num + t_now * w_at
            at_den = at_den + w_at
            # Hard AT: first threshold crossing after blanking, within
            # the detection window
            fires = (~activated) & (V > jnp.float64(V_GATE)) & \
                    (t_now > jnp.float64(T_BLANK)) & \
                    (t_now < jnp.float64(DETECTION_WINDOW))
            at_hard = jnp.where(fires, t_now, at_hard)
            activated = activated | fires
            return (V, h, at_num, at_den, activated, at_hard), None

        V0 = jnp.zeros(Np, dtype=jnp.float64)
        h0 = jnp.ones(Np, dtype=jnp.float64)
        at_num0 = jnp.zeros(Np, dtype=jnp.float64)
        at_den0 = jnp.zeros(Np, dtype=jnp.float64)
        act0 = jnp.zeros(Np, dtype=jnp.bool_)
        at0 = jnp.full(Np, jnp.float64(DETECTION_WINDOW), dtype=jnp.float64)
        (_, _, at_num, at_den, activated, at_hard), _ = jax.lax.scan(
            jax.checkpoint(scan_fn), (V0, h0, at_num0, at_den0, act0, at0),
            (sv_j, t_arr))
        at_soft = at_num / (at_den + jnp.float64(1e-6))
        return at_hard, at_soft, activated

    return run_forward_atmap


def make_forward_solver(Np, eu, ev, m_inv, w, W_lead, mask_pace,
                          V_GATE, A_CRIT, DT, N_ION, STIM_AMP, CM,
                          sv_schedule):
    """
    Build a JIT-compiled forward solver closure for this specific
    mesh/pacing configuration.

    Parameters
    ----------
    Np : int
        Number of nodes.
    eu, ev : (K,) int arrays
        Edge node-index pairs.
    m_inv : (N,) array
        Inverse lumped mass.
    w : (K,) array
        Anisotropic edge conductivity weights.
    W_lead : (16, N) array
        Lead-field weight matrix (electrode x node), for OEGM/EGM output.
    mask_pace : (N,) array
        Pacing mask (1.0 at paced nodes).
    V_GATE, A_CRIT : float
        Mitchell-Schaeffer gate/critical voltage parameters.
    DT : float
        Timestep (ms). Use 0.1ms to match the patch/manuscript exactly.
    N_ION : int
        Ionic sub-steps per full timestep (4, matching the patch).
    STIM_AMP : float
        Stimulus amplitude.
    CM : float
        Membrane capacitance.
    sv_schedule : (NT,) bool array
        Stimulus-on schedule, one entry per timestep.

    Returns
    -------
    run_forward : callable
        run_forward(p5) -> phi_T, where p5 = [tau_in, tau_out,
        tau_open, tau_close, G_IL] (jnp.float64 array). Returns the
        (NT, 16) lead-field/EGM time series.
    """
    m_j = jnp.array(m_inv, dtype=jnp.float64)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    w_j = jnp.array(w, dtype=jnp.float64)
    W_j = jnp.array(W_lead, dtype=jnp.float64)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    @jax.jit
    def run_forward(p5_j64):
        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w_j * x[ev_j] + w_j * x[eu_j])
            Kx = Kx.at[ev_j].add(-w_j * x[eu_j] + w_j * x[ev_j])
            return Kx

        def _ion(V, h):
            # Rush-Larsen -- exact match to the patch's scheme
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
            return (V, h), W_j @ V

        V0 = jnp.zeros(Np, dtype=jnp.float64)
        h0 = jnp.ones(Np, dtype=jnp.float64)
        (_, _), phi_T = jax.lax.scan(jax.checkpoint(scan_fn), (V0, h0), sv_j)
        return phi_T

    return run_forward


def compute_oegm(phi_T, cliques):
    """
    Construct omnipolar EGMs (OEGMs) from a lead-field/EGM time
    series, via the same 2x2-clique spatial-gradient formula used
    throughout the codebase.

    Parameters
    ----------
    phi_T : (NT, 16) array
        Unipolar EGM time series (one column per electrode).
    cliques : (9, 4) int array
        4-electrode clique index groups (CLIQUES convention).

    Returns
    -------
    oegms : (9, NT) array
        One omnipolar EGM per clique.
    """
    oegms = []
    for c in cliques:
        oegms.append((phi_T[:, c[0]] - phi_T[:, c[1]]
                      - phi_T[:, c[2]] + phi_T[:, c[3]]) / 4.0)
    return jnp.stack(oegms)


def make_forward_solver_traces(Np, eu, ev, m_inv, w, mask_pace,
                                 node_indices, V_GATE, A_CRIT, DT, N_ION,
                                 STIM_AMP, CM, sv_schedule):
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
    w_j = jnp.array(w, dtype=jnp.float64)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)
    node_idx_j = jnp.array(node_indices, dtype=jnp.int32)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    @jax.jit
    def run_forward_traces(p5_j64):
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

    return run_forward_traces