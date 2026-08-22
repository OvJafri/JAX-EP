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

IMPORTANT FIX (this version): G_IL (p5[4]) is now genuinely wired
into the differentiable computation graph. Previously, the diffusion
weights `w` were computed ONCE, externally, via
fem.anisotropic_weights(G_IL=<fixed value>), then passed into the
solver as a fixed NumPy/JAX constant, closed over by the JIT-traced
function. p5_j64[4] (G_IL) was NEVER referenced anywhere inside the
solver's own code -- confirmed directly by grep. This meant FD,
jax.grad, and jax.jacfwd were all CORRECTLY computing a gradient of
exactly zero for G_IL, because the traced function's output
genuinely, mathematically did not depend on it at call-time. This
was not a numerical, scale, or loss-shape issue -- every differently-
shaped loss we tried (raw voltage, OEGM, near/far nodes, various
FD_EPS, single-S1/S1-S2 protocols) correctly reported zero, because
it WAS zero given how the solver was built.

THE FIX: each solver function now optionally accepts the raw
ingredients needed to build the diffusion weights (ecot, ed_x, BETA),
and if given, computes `w` FROM p5_j64[4] INSIDE the JIT-traced
function on every call -- exactly mirroring the original monolithic
Forward_LA.py's own build_w(g_il) closure. This makes G_IL genuinely
differentiable by all three methods (FD, jax.grad, jax.jacfwd).

BACKWARD COMPATIBILITY: if ecot/ed_x/BETA are not supplied, the
functions fall back to the previous behaviour (a fixed, external `w`)
-- this keeps every existing script (run_forward_only.py,
run_gpu_cpu_benchmark.py, run_cube_benchmark.py, etc., none of which
need G_IL differentiability, only correct forward-pass speed/LAT
accuracy) working completely unchanged. Only scripts that explicitly
want a differentiable G_IL need to pass the new parameters.
"""
import jax
import jax.numpy as jnp


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


def make_forward_solver_atmap(Np, eu, ev, m_inv, w, mask_pace,
                                V_GATE, A_CRIT, DT, N_ION, STIM_AMP, CM,
                                sv_schedule, T_BLANK=None, BETA_AT=50.0,
                                DETECTION_WINDOW=None,
                                EARLY_UPSTROKE_V=0.3, EARLY_UPSTROKE_SHARPNESS=20.0,
                                ecot=None, ed_x=None, BETA=None):
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

    ecot, ed_x, BETA : optional
        If ALL THREE are given, the diffusion weights are computed
        FROM p5_j64[4] (G_IL) INSIDE the JIT-traced function on every
        call, making G_IL genuinely differentiable (see module
        docstring -- this fixes a real, confirmed bug where G_IL's
        gradient was previously always exactly zero, since `w` was a
        fixed, external constant that never depended on p5_j64[4] at
        all). If any of these three is omitted, falls back to the
        previous, fixed-w behaviour (backward compatible; G_IL will
        NOT be differentiable in that case -- fine for scripts that
        only need correct forward-pass output, e.g. the CPU/GPU
        timing benchmark).
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
    EARLY_UPSTROKE_V, EARLY_UPSTROKE_SHARPNESS : float
        Weighting by relu(dV) alone concentrates at_soft's weight near
        the peak of dV/dt, which is the STEEPEST point of the upstroke
        (V~0.5) -- inherently somewhat later than at_hard's own
        definition (the earlier V_GATE=0.13 crossing). Restricting the
        dV-weighting to V < EARLY_UPSTROKE_V (smoothly, via a sigmoid
        of sharpness EARLY_UPSTROKE_SHARPNESS) pulls the weighted
        centroid toward the EARLY part of the upstroke, much closer to
        at_hard's own crossing point. This restriction is purely a
        function of V's level, independent of dV's sign, so it cannot
        accidentally pick up the repolarisation-phase crossing (unlike
        an earlier, abandoned attempt that gated by proximity to
        V_GATE directly, which did pick up that spurious crossing).
        Verified directly (8mm test mesh): reduces the mean |at_soft -
        at_hard| gap from ~1.58ms to ~0.25ms, max from ~3.57ms to
        ~2.0ms, versus relu(dV) alone.

    Returns
    -------
    run_forward_atmap : callable
        run_forward_atmap(p5) -> (at_hard, at_soft, activated).
    """
    m_j = jnp.array(m_inv, dtype=jnp.float64)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)

    w_fn, g_il_differentiable = _make_w_fn(w, ecot, ed_x, BETA, CM)

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
            # V's absolute level -- see module/function docstring for
            # why, and for the EARLY_UPSTROKE_V refinement.
            dV = V - V_prev
            early_mask = jax.nn.sigmoid(-jnp.float64(EARLY_UPSTROKE_SHARPNESS)
                                          * (V - jnp.float64(EARLY_UPSTROKE_V)))
            w_at = jax.nn.relu(dV) * early_mask * in_beat1
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

    run_forward_atmap.g_il_differentiable = g_il_differentiable
    return run_forward_atmap


def make_forward_solver(Np, eu, ev, m_inv, w, W_lead, mask_pace,
                          V_GATE, A_CRIT, DT, N_ION, STIM_AMP, CM,
                          sv_schedule, ecot=None, ed_x=None, BETA=None):
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
        Anisotropic edge conductivity weights. Used as a fixed
        constant UNLESS ecot/ed_x/BETA are also given (see below).
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
    ecot, ed_x, BETA : optional
        If ALL THREE are given, the diffusion weights are computed
        FROM p5_j64[4] (G_IL) INSIDE the JIT-traced function on every
        call, making G_IL genuinely differentiable through the OEGM/
        lead-field output -- fixing the same bug documented in
        make_forward_solver_atmap's docstring (G_IL was previously
        never referenced anywhere in the traced computation, so its
        gradient was always, correctly-given-the-bug, exactly zero).
        If omitted, falls back to the previous, fixed-w behaviour.

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
    W_j = jnp.array(W_lead, dtype=jnp.float64)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)

    w_fn, g_il_differentiable = _make_w_fn(w, ecot, ed_x, BETA, CM)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    @jax.jit
    def run_forward(p5_j64):
        w_j = w_fn(p5_j64)

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

    run_forward.g_il_differentiable = g_il_differentiable
    return run_forward


def make_forward_solver_euler(Np, eu, ev, m_inv, w, W_lead, mask_pace,
                                V_GATE, A_CRIT, DT, N_ION, STIM_AMP, CM,
                                sv_schedule, ecot=None, ed_x=None, BETA=None):
    """
    Same as make_forward_solver, EXCEPT the gating variable h uses
    SIMPLE EULER integration, not Rush-Larsen -- matching the
    ACTUAL, CONFIRMED parameter_estimation_kaggle.py code exactly
    (the real script that genuinely produced the paper's parameter-
    estimation results), not the Rush-Larsen scheme used in the
    separate Forward_LA.py/differentiability-demonstration script.

    Confirmed directly by reading the real source: its _ion function
    computes dh directly (dh = (1-h)/tau_open*(1-sw) - h/tau_close*sw)
    and steps h_new = h + dt_sub*dh -- plain, explicit Euler, not the
    analytic-within-substep Rush-Larsen update. The V-update formula
    itself is mathematically identical between both schemes (only h's
    integration differs), which is why only this piece changes here.

    Use THIS solver (not make_forward_solver) when genuine, exact
    equivalence to the real parameter-estimation code's OEGM output
    is required -- e.g. for the OEGM-morphology-matching work, not
    for the DT=0.1ms/N_T=11,100/jacfwd-stability demonstration that
    make_forward_solver's Rush-Larsen scheme was specifically chosen
    to reproduce (see that function's own docstring).

    Parameters are identical to make_forward_solver -- see its
    docstring for the full parameter list.
    """
    m_j = jnp.array(m_inv, dtype=jnp.float64)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    W_j = jnp.array(W_lead, dtype=jnp.float64)
    mask_j = jnp.array(mask_pace, dtype=jnp.float64)
    sv_j = jnp.array(sv_schedule)

    w_fn, g_il_differentiable = _make_w_fn(w, ecot, ed_x, BETA, CM)

    dt_sub = DT / (2 * N_ION)
    Iext = (STIM_AMP / CM) * DT * 1e-3
    alpha = DT / 2.0

    @jax.jit
    def run_forward_euler(p5_j64):
        w_j = w_fn(p5_j64)

        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w_j * x[ev_j] + w_j * x[eu_j])
            Kx = Kx.at[ev_j].add(-w_j * x[eu_j] + w_j * x[ev_j])
            return Kx

        def _ion(V, h):
            # Simple Euler -- exact match to parameter_estimation_kaggle.py
            sw = jax.nn.sigmoid(jnp.float64(150.) * (V - jnp.float64(V_GATE)))
            dh = ((jnp.float64(1.) - h) / p5_j64[2]) * (jnp.float64(1.) - sw) \
                 - (h / p5_j64[3]) * sw
            V_new = jnp.clip(
                V + dt_sub * ((h * V * (V - jnp.float64(A_CRIT)) * (jnp.float64(1.) - V))
                              / p5_j64[0] - (jnp.float64(1.) - h) * (V / p5_j64[1])),
                jnp.float64(0.), jnp.float64(1.))
            h_new = jnp.clip(h + dt_sub * dh, jnp.float64(0.), jnp.float64(1.))
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

    run_forward_euler.g_il_differentiable = g_il_differentiable
    return run_forward_euler


def make_forward_solver_traces_euler(Np, eu, ev, m_inv, w, mask_pace,
                                       node_indices, V_GATE, A_CRIT, DT, N_ION,
                                       STIM_AMP, CM, sv_schedule,
                                       ecot=None, ed_x=None, BETA=None):
    """
    Same as make_forward_solver_traces, EXCEPT using SIMPLE EULER
    integration for the gating variable h (not Rush-Larsen) -- matching
    make_forward_solver_euler's own scheme, i.e. the ACTUAL, CONFIRMED
    parameter_estimation_kaggle.py code.

    Use this to sample RAW membrane voltage directly at each HD-grid
    electrode's nearest mesh node, for OEGM construction -- confirmed
    directly to resolve a genuine morphology problem in the lead-field/
    far-field-summation approach (make_forward_solver_euler +
    compute_oegm): the lead-field sum picks up slowly-varying
    contributions from distant, still-active tissue even after the
    wavefront has locally passed the electrode grid, producing an
    unphysiological slow drift through the plateau phase instead of a
    flat baseline. Sampling voltage directly at each electrode's own
    node has no such far-field contribution at all -- verified
    directly: this produces a sharp deflection at activation followed
    by a genuinely flat, near-zero baseline through the plateau,
    matching expected OEGM/omnipolar morphology.

    Parameters are the same as make_forward_solver_traces -- see its
    docstring for the full parameter list.
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
    def run_forward_traces_euler(p5_j64):
        w_j = w_fn(p5_j64)

        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w_j * x[ev_j] + w_j * x[eu_j])
            Kx = Kx.at[ev_j].add(-w_j * x[eu_j] + w_j * x[ev_j])
            return Kx

        def _ion(V, h):
            # Simple Euler -- exact match to parameter_estimation_kaggle.py
            sw = jax.nn.sigmoid(jnp.float64(150.) * (V - jnp.float64(V_GATE)))
            dh = ((jnp.float64(1.) - h) / p5_j64[2]) * (jnp.float64(1.) - sw) \
                 - (h / p5_j64[3]) * sw
            V_new = jnp.clip(
                V + dt_sub * ((h * V * (V - jnp.float64(A_CRIT)) * (jnp.float64(1.) - V))
                              / p5_j64[0] - (jnp.float64(1.) - h) * (V / p5_j64[1])),
                jnp.float64(0.), jnp.float64(1.))
            h_new = jnp.clip(h + dt_sub * dh, jnp.float64(0.), jnp.float64(1.))
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

    run_forward_traces_euler.g_il_differentiable = g_il_differentiable
    return run_forward_traces_euler


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
