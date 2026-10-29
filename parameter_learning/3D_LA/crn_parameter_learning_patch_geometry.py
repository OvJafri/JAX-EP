# -*- coding: utf-8 -*-
"""
crn_parameter_learning_patch_geometry.py
=========================================
CRN (Courtemanche-Ramirez-Nattel 1998) parameter learning on the LA patch, from simulated
omnipolar EGMs, with the differentiable JAX version of the CRN model.  CRN counterpart of
parameter_learning/3D_LA/parameter_learning_patch_geometry.py (mMS).

Recovered parameters (6): [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL]
  m_* multiply the nominal CRN conductances; G_IL in mS/cm.

Pipeline (as in the paper)
  Phase 1: Nelder-Mead (with restarts), 9-clique S2 EGM loss
  Phase 2: L-BFGS-B with exact forward-mode gradients (jax.jacfwd), 9-clique S1 + 0.5 x S2 loss
  Differentiable CRN: sigmoid -40 mV switch in the h/j gates (CRN_SMOOTH_MV = 0.5 mV) and the
  removable 0/0 points of the rate functions evaluated by their exact limits (CRN_SAFE_POLES = 1).

Inputs (no clinical mesh is needed)
  ../patch_geometry.npz                                LA patch (2,894 nodes), HD grid, fibres, pacing edge (shared with mMS)
  <GT_DIR>/egm_pipeline_for_jax_{case}/gt_crn_egm_{case}.npz     target EGMs (no parameters inside)
  <GT_DIR>/egm_pipeline_for_jax_{case}/gt_crn_truth_{case}.json  optional; used only AFTER the fit for MAPE

Run from parameter_learning/3D_LA/CRN:   python crn_parameter_learning_patch_geometry.py

Outputs (WORK_DIR)
  results_crn_{case}_*.npz, crn_fit_{case}_*.png, p1_cache_crn_{case}_*.npz

Settings can be overridden with environment variables (PATCH_GEOMETRY_PATH, GT_DIR, WORK_DIR, CASES,
GRAD_MODE, CRN_SMOOTH_MV, CRN_SAFE_POLES, ...); the defaults reproduce the paper.
"""
import os, sys, time, json
os.environ.setdefault("MPLBACKEND", "Agg")   # headless node
import numpy as np
import jax, jax.numpy as jnp
from collections import defaultdict

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64
NP_DTYPE = np.float64

# ══════════════════════════════════════════════════════════════════════════
# [0] PATHS + TRUTH
# ══════════════════════════════════════════════════════════════════════════
PATCH_GEOMETRY_PATH = os.environ.get("PATCH_GEOMETRY_PATH", "../patch_geometry.npz")   # 3D_LA/patch_geometry.npz (run from 3D_LA/CRN)
PARAM_NAMES = ['m_gNa', 'm_gto', 'm_gKr', 'm_gCaL', 'm_gK1', 'G_IL']
CASES = os.environ.get("CASES", "set1,set5,set2").split(",")

# ── Physics (same as mMS GT script) ─────────────────────────────────────────
BETA = float(os.environ.get("BETA", 140.)); CM = 1.     # beta = 140 (as in the CRN Colab script; mMS GT used 100)
STIM_AMP = 20.0            # pA/pF  (CRN units; same as Colab forward script)
STIM_DUR = 2.
# ---- pacing protocol: THREE sequences per case, each = S1, S1, S2 (S2 coupling interval 450, 400, 350 ms) --------
# The three sequences are fitted simultaneously: they are simulated as one jax.vmap batch (same parameters, different stimulus schedule).
S1_START = 10.
S1_BCL = float(os.environ.get("S1_BCL", 500.))
N_COND = int(os.environ.get("N_COND", 1))
CI_LIST = [float(x) for x in os.environ.get("CI_LIST", "500").split(",")]
S2_WIN_MS = 400.
def _prot(ci):
    st = [S1_START + k*S1_BCL for k in range(N_COND)]; st.append(st[-1] + ci)
    return dict(ci=ci, stim=st, s2_on=st[-1], beat_start=list(st),
                beat_len=[st[k+1]-st[k] for k in range(len(st)-1)] + [S2_WIN_MS])
PROTS = [_prot(ci) for ci in CI_LIST]
TOTAL_MS = float(os.environ.get("TOTAL_MS_OVERRIDE", max(p['s2_on'] for p in PROTS) + 600.))   # same length for all sequences

DT_SIM = 0.1
N_ION = int(os.environ.get("N_ION", 4))
CG_TOL = float(os.environ.get("CG_TOL", 1e-6)); CG_MAXITER = int(os.environ.get("CG_MAXITER", 50))

V_SCALE = 100.; V_SHIFT = -81.     # V_norm = (V - V_SHIFT)/V_SCALE   (fixed, not fitted)

CLIQUES = np.array([[1,2,5,6],[2,3,6,7],[3,4,7,8],
                    [5,6,9,10],[6,7,10,11],[7,8,11,12],
                    [9,10,13,14],[10,11,14,15],[11,12,15,16]]) - 1
CENTRE_CLIQUE = 4
EDGE_ELEC = {"LEFT": np.array([0,4,8,12]), "RIGHT": np.array([3,7,11,15]),
             "BOTTOM": np.array([0,1,2,3]), "TOP": np.array([12,13,14,15])}
OFFSET_MM = 3.0; STRIP_PAD_MM = 2.0; BB_PAD = 5000.

# ── CRN constants (from la_forward_courtemanche_colab.py) ───────────────────
R = 8.3143; T = 310.; F = 96.4867; C_m = 100.
V_i = 13668.; V_up = 1109.52; V_rel = 96.48
C_Fn1 = 9.648e-13; C_Fn2 = 2.5910306809e-13
C_B1a = 3.79138232501097e-05; C_B1b = 0.0811764705882353
C_B1c = 0.00705882352941176; C_dCaup = 0.0869565217391304
C_dCa_rel = 8.0; C_B1d = 0.00537112496043221; C_B1e = 11.5
C_Iupl = 0.000333333333333
KmCsqn = 0.8; maxCsqn = 10.; KmCmdn = 0.00238; maxCmdn = 0.05
KmTrpn = 0.0005; maxTrpn = 0.07
# nominal conductances (original CRN 1998 values)
NOM = dict(g_Na=7.8, g_to=0.1652, g_Kr=0.0294, g_CaL=0.1238, g_K1=0.09, g_Ks=0.129)
g_bCa = 0.00113; g_bNa = 0.000674
K_o = 5.4; Na_o = 140.; Ca_o = 1.8; Na_i = 11.2; K_i = 139.
maxI_NaCa = 1600.; maxI_pCa = 0.275; maxI_NaK = 0.60
maxI_up = 0.005; k_rel = 30.; K_up = 0.00092; tau_tr = 180.
gamma_ = 0.35; KmNa_i = 10.; KmK_o = 1.5; KmNa = 87.5
KmCa = 1.38; k_sat = 0.1; K_Q10 = 3.0
# ── Optional smoothing of the two non-smooth parts of CRN (defaults = published model) ──────────
#   CRN_SMOOTH_MV > 0 : the hard -40 mV branch switch in the I_Na h and j gates becomes a sigmoid blend of the
#                       two published branches, width CRN_SMOOTH_MV (mV).  0 = published jnp.where switch.
#   CRN_TRIG_W        : multiplies the width (13.67e-16) of the SR-release logistic triggers (u_inf, v_inf, tau_v).  1 = published.
#   Single-cell test (S1-S2 500->300 ms): SMOOTH_MV=0.5, TRIG_W=10 changes APD90 by <= 0.2 ms and makes jacfwd agree
#   with central FD at optimiser-scale steps.
SMOOTH_MV = float(os.environ.get("CRN_SMOOTH_MV", 0.5))     # paper: 0.5 mV
TRIG_W    = float(os.environ.get("CRN_TRIG_W", 1.))
#   CRN_SAFE_POLES=1  : evaluate the removable singularities of the published rate functions (a_m, xr, xs, tau_d, tau_w)
#                       with their exact limits instead of "+1e-8"-type guards.  The guards create a true pole a hair's breadth
#                       from each singular voltage (e.g. tau_w at V = 7.9 - 1e-4 mV); a trajectory passing close to it gives a spike
#                       that makes the loss rough on very small scales.  Same mathematical model, only safer numerics.
SAFE_POLES = os.environ.get("CRN_SAFE_POLES", "1") == "1"  # paper: exact limits
SMOOTH_TAG = ("" if (SMOOTH_MV == 0. and TRIG_W == 1.) else f"_smooth{SMOOTH_MV:g}mV_trig{TRIG_W:g}") + ("_safepoles" if SAFE_POLES else "")

def _xexp(x, k):
    """x / (1 - exp(-x/k)), with its exact limit k at x = 0 (double-where: finite value AND finite tangent everywhere)."""
    small = jnp.abs(x) < 1e-6
    xs = jnp.where(small, jnp.ones_like(x), x)
    return jnp.where(small, k + x/2., xs/(-jnp.expm1(-xs/k)))
E_Na_ = R*T/F*np.log(Na_o/Na_i)
E_K_  = R*T/F*np.log(K_o/K_i)
sigma_ = (np.exp(Na_o/67.3) - 1.)/7.

# ══════════════════════════════════════════════════════════════════════════
# [1] CRN IONIC MODEL  (equations verbatim from la_forward_courtemanche_colab.py;
#     only change: conductances are arguments)
# ══════════════════════════════════════════════════════════════════════════
def crn_initial(Np_):
    """Spatially uniform resting state (tuple of 19 arrays, V in mV)."""
    return (jnp.full(Np_, DTYPE(-81.2)), jnp.full(Np_, DTYPE(2.91e-3)),
            jnp.full(Np_, DTYPE(0.965)), jnp.full(Np_, DTYPE(0.978)),
            jnp.full(Np_, DTYPE(3.04e-2)), jnp.full(Np_, DTYPE(0.999)),
            jnp.full(Np_, DTYPE(4.96e-3)), jnp.full(Np_, DTYPE(0.999)),
            jnp.full(Np_, DTYPE(3.29e-5)), jnp.full(Np_, DTYPE(1.87e-2)),
            jnp.full(Np_, DTYPE(1.37e-4)), jnp.full(Np_, DTYPE(0.999)),
            jnp.full(Np_, DTYPE(0.775)), jnp.zeros(Np_, dtype=DTYPE),
            jnp.ones(Np_, dtype=DTYPE), jnp.full(Np_, DTYPE(0.999)),
            jnp.full(Np_, DTYPE(1.02e-1)), jnp.full(Np_, DTYPE(1.49)),
            jnp.full(Np_, DTYPE(1.49)))

def crn_ion_step(state_tuple, I_stim_vec, dt, gNa, gto, gKr, gCaL, gK1, gKs):
    """One ionic timestep using Rush-Larsen for all gates.
    state_tuple: tuple of 19 arrays each (Np,)
    I_stim_vec:  (Np,) stimulus current pA/pF
    dt:          timestep ms
    Returns: new state_tuple
    """
    (V,m,h,j,oa,oi,ua,ui,xr,xs,d,f,f_Ca,u,vv,w,Ca_i,Ca_up,Ca_rel) = state_tuple

    E_Na = DTYPE(E_Na_); E_K = DTYPE(E_K_); sigma = DTYPE(sigma_)
    conCa = Ca_i / DTYPE(1000.)  # umol/L → mmol/L

    def rl(x, x_inf, tau):
        return x_inf + (x-x_inf)*jnp.exp(-dt/(tau+DTYPE(1e-10)))

    # m gate
    if SAFE_POLES:
        a_m = DTYPE(0.32)*_xexp(V+DTYPE(47.13), DTYPE(10.))
    else:
        a_m = jnp.where(jnp.abs(V+DTYPE(47.13))<DTYPE(1e-3), DTYPE(3.2),
                    DTYPE(0.32)*(V+DTYPE(47.13))/(DTYPE(1.)-jnp.exp(DTYPE(-0.1)*(V+DTYPE(47.13)))))
    b_m = DTYPE(0.08)*jnp.exp(-V/DTYPE(11.))
    m_inf = a_m/(a_m+b_m+DTYPE(1e-10)); tau_m = DTYPE(1.)/(a_m+b_m+DTYPE(1e-10))

    # h and j gates: published branches above (hi) and below (lo) -40 mV
    Vlo = jnp.minimum(V, DTYPE(0.))     # lower-branch formulas are only ever weighted for V < -40; clamp keeps their exp() finite
    a_h_hi = DTYPE(0.)
    a_h_lo = DTYPE(0.135)*jnp.exp((Vlo+DTYPE(80.))/DTYPE(-6.8))
    b_h_hi = DTYPE(1./0.13)/(DTYPE(1.)+jnp.exp(-(V+DTYPE(10.66))/DTYPE(11.1)))
    b_h_lo = DTYPE(3.56)*jnp.exp(DTYPE(0.079)*Vlo)+DTYPE(3.1e5)*jnp.exp(DTYPE(0.35)*Vlo)
    a_j_hi = DTYPE(0.)
    a_j_lo = (DTYPE(-127140.)*jnp.exp(DTYPE(0.2444)*Vlo)-DTYPE(3.474e-5)*jnp.exp(DTYPE(-0.04391)*Vlo)) \
             *(Vlo+DTYPE(37.78))/(DTYPE(1.)+jnp.exp(DTYPE(0.311)*(Vlo+DTYPE(79.23))))
    b_j_hi = DTYPE(0.3)*jnp.exp(DTYPE(-2.535e-7)*V)/(DTYPE(1.)+jnp.exp(DTYPE(-0.1)*(V+DTYPE(32.))))
    b_j_lo = DTYPE(0.1212)*jnp.exp(DTYPE(-0.01052)*V)/(DTYPE(1.)+jnp.exp(DTYPE(-0.1378)*(V+DTYPE(40.14))))
    if SMOOTH_MV > 0.:
        s_hi = jax.nn.sigmoid((V+DTYPE(40.))/DTYPE(SMOOTH_MV)); s_lo = jax.nn.sigmoid(-(V+DTYPE(40.))/DTYPE(SMOOTH_MV))
        a_h = s_hi*a_h_hi + s_lo*a_h_lo; b_h = s_hi*b_h_hi + s_lo*b_h_lo
        a_j = s_hi*a_j_hi + s_lo*a_j_lo; b_j = s_hi*b_j_hi + s_lo*b_j_lo
    else:                                 # published: hard switch (h: V >= -40 upper branch; j: V < -40 lower branch)
        hi = V >= DTYPE(-40.)
        a_h = jnp.where(hi, a_h_hi, a_h_lo); b_h = jnp.where(hi, b_h_hi, b_h_lo)
        a_j = jnp.where(hi, a_j_hi, a_j_lo); b_j = jnp.where(hi, b_j_hi, b_j_lo)
    h_inf = a_h/(a_h+b_h+DTYPE(1e-10)); tau_h = DTYPE(1.)/(a_h+b_h+DTYPE(1e-10))
    j_inf = a_j/(a_j+b_j+DTYPE(1e-10)); tau_j = DTYPE(1.)/(a_j+b_j+DTYPE(1e-10))

    # oa gate
    aa_oa = DTYPE(0.65)/(jnp.exp((V+DTYPE(10.))/DTYPE(-8.5))+jnp.exp((DTYPE(30.)-V)/DTYPE(59.)))
    bb_oa = DTYPE(0.65)/(DTYPE(2.5)+jnp.exp((V+DTYPE(82.))/DTYPE(17.)))
    tau_oa = DTYPE(1.)/(aa_oa+bb_oa)/DTYPE(K_Q10)
    oa_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp(-(V+DTYPE(20.47))/DTYPE(17.54)))

    # oi gate
    aa_oi = DTYPE(1.)/(DTYPE(18.53)+jnp.exp((V+DTYPE(113.7))/DTYPE(10.95)))
    bb_oi = DTYPE(1.)/(DTYPE(35.56)+jnp.exp(-(V+DTYPE(1.26))/DTYPE(7.44)))
    tau_oi = DTYPE(1.)/(aa_oi+bb_oi)/DTYPE(K_Q10)
    oi_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp((V+DTYPE(43.1))/DTYPE(5.3)))

    # ua gate
    aa_ua = DTYPE(0.65)/(jnp.exp((V+DTYPE(10.))/DTYPE(-8.5))+jnp.exp((V-DTYPE(30.))/DTYPE(-59.)))
    bb_ua = DTYPE(0.65)/(DTYPE(2.5)+jnp.exp((V+DTYPE(82.))/DTYPE(17.)))
    tau_ua = DTYPE(1.)/(aa_ua+bb_ua)/DTYPE(K_Q10)
    ua_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp((V+DTYPE(30.3))/DTYPE(-9.6)))

    # ui gate
    aa_ui = DTYPE(1.)/(DTYPE(21.)+jnp.exp((V-DTYPE(185.))/DTYPE(-28.)))
    bb_ui = jnp.exp((V-DTYPE(158.))/DTYPE(16.))
    tau_ui = DTYPE(1.)/(aa_ui+bb_ui)/DTYPE(K_Q10)
    ui_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp((V-DTYPE(99.45))/DTYPE(27.48)))

    # xr gate
    if SAFE_POLES:
        aa_xr = DTYPE(0.0003)*_xexp(V+DTYPE(14.1), DTYPE(5.))
        bb_xr = DTYPE(7.3898e-5)*_xexp(-(V-DTYPE(3.3328)), DTYPE(5.1237))      # y/(e^{y/k}-1) = (-y)/(1-e^{-(-y)/k})
    else:
        aa_xr = DTYPE(0.0003)*(V+DTYPE(14.1))/(DTYPE(1.)-jnp.exp((V+DTYPE(14.1))/DTYPE(-5.))+DTYPE(1e-8))
        bb_xr = DTYPE(7.3898e-5)*(V-DTYPE(3.3328))/(jnp.exp((V-DTYPE(3.3328))/DTYPE(5.1237))-DTYPE(1.)+DTYPE(1e-8))
    tau_xr = DTYPE(1.)/(aa_xr+bb_xr+DTYPE(1e-8))
    xr_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp((V+DTYPE(14.1))/DTYPE(-6.5)))

    # xs gate
    if SAFE_POLES:
        aa_xs = DTYPE(4e-5)*_xexp(V-DTYPE(19.9), DTYPE(17.))
        bb_xs = DTYPE(3.5e-5)*_xexp(-(V-DTYPE(19.9)), DTYPE(9.))
    else:
        aa_xs = DTYPE(4e-5)*(V-DTYPE(19.9))/(DTYPE(1.)-jnp.exp((DTYPE(19.9)-V)/DTYPE(17.))+DTYPE(1e-8))
        bb_xs = DTYPE(3.5e-5)*(V-DTYPE(19.9))/(jnp.exp((V-DTYPE(19.9))/DTYPE(9.))-DTYPE(1.)+DTYPE(1e-8))
    tau_xs = DTYPE(0.5)/(aa_xs+bb_xs+DTYPE(1e-8))
    xs_inf = DTYPE(1.)/jnp.sqrt(DTYPE(1.)+jnp.exp((V-DTYPE(19.9))/DTYPE(-12.7)))

    # d gate
    if SAFE_POLES:     # tau_d = (1-e^{-x/6.24}) / (0.035 x (1+e^{-x/6.24})) = 1 / (0.035 * xexp(x, 6.24) * (1+e^{-x/6.24}))
        _xd = V+DTYPE(10.)
        tau_d = DTYPE(1.)/(DTYPE(0.035)*_xexp(_xd, DTYPE(6.24))*(DTYPE(1.)+jnp.exp(-_xd/DTYPE(6.24))))
    else:
        tau_d = jnp.where(jnp.abs(V+DTYPE(10.))<DTYPE(1e-3),
                      DTYPE(1./6.24/0.035/2.),
                      (DTYPE(1.)-jnp.exp(-(V+DTYPE(10.))/DTYPE(6.24)))
                      /(DTYPE(0.035)*(V+DTYPE(10.)+DTYPE(1e-6))*(DTYPE(1.)+jnp.exp(-(V+DTYPE(10.))/DTYPE(6.24)))))
    d_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp(-(V+DTYPE(10.))/DTYPE(8.)))

    # f gate
    tau_f = DTYPE(9.)/(DTYPE(0.0197)*jnp.exp(-(DTYPE(0.0337)*(V+DTYPE(10.)))**2)+DTYPE(0.02))
    f_inf = DTYPE(1.)/(DTYPE(1.)+jnp.exp((V+DTYPE(28.))/DTYPE(6.9)))

    # f_Ca gate
    tau_fCa = DTYPE(2.)
    f_Ca_inf = DTYPE(1.)/(DTYPE(1.)+conCa/DTYPE(0.00035))

    # w gate
    if SAFE_POLES:     # tau_w = 6 (1-e^{-y/5}) / ((1+0.3 e^{-y/5}) y) = 6 / ((1+0.3 e^{-y/5}) * xexp(y, 5))
        _yw = V-DTYPE(7.9)
        tau_w = DTYPE(6.)/((DTYPE(1.)+DTYPE(0.3)*jnp.exp(-_yw/DTYPE(5.)))*_xexp(_yw, DTYPE(5.)))
    else:
        tau_w = DTYPE(6.)*(DTYPE(1.)-jnp.exp((DTYPE(7.9)-V)/DTYPE(5.))) \
            /(DTYPE(1.)+DTYPE(0.3)*jnp.exp((DTYPE(7.9)-V)/DTYPE(5.)))/(V-DTYPE(7.9)+DTYPE(1e-4))
    w_inf = DTYPE(1.)-DTYPE(1.)/(DTYPE(1.)+jnp.exp((DTYPE(40.)-V)/DTYPE(17.)))

    # Currents (pA/pF)
    i_Na  = gNa*m**3*h*j*(V-E_Na)
    i_K1  = gK1*(V-E_K)/(DTYPE(1.)+jnp.exp(DTYPE(0.07)*(V+DTYPE(80.))))
    i_to  = gto*oa**3*oi*(V-E_K)
    g_Kur = DTYPE(0.005)+DTYPE(0.05)/(DTYPE(1.)+jnp.exp((DTYPE(15.)-V)/DTYPE(13.)))
    i_Kur = g_Kur*ua**3*ui*(V-E_K)
    i_Kr  = gKr*xr*(V-E_K)/(DTYPE(1.)+jnp.exp((V+DTYPE(15.))/DTYPE(22.4)))
    i_Ks  = gKs*xs**2*(V-E_K)
    i_CaL = gCaL*d*f*f_Ca*(V-DTYPE(65.))

    f_NaK = DTYPE(1.)/(DTYPE(1.)+DTYPE(0.1245)*jnp.exp(DTYPE(-0.1*F/R/T)*V)
                       +DTYPE(0.0365)*sigma*jnp.exp(DTYPE(-F/R/T)*V))
    i_NaK = DTYPE(maxI_NaK)*f_NaK/(DTYPE(1.)+(DTYPE(KmNa_i/Na_i))**DTYPE(1.5)) \
            *DTYPE(K_o)/(DTYPE(K_o)+DTYPE(KmK_o))

    KmNa3=DTYPE(KmNa**3); Na_o3=DTYPE(Na_o**3)
    denom_NaCa=(KmNa3+Na_o3)*(DTYPE(KmCa)+DTYPE(Ca_o)) \
               *(DTYPE(1.)+DTYPE(k_sat)*jnp.exp(DTYPE((gamma_-1.)*F/R/T)*V))
    i_NaCa=DTYPE(maxI_NaCa)*(jnp.exp(DTYPE(gamma_*F/R/T)*V)*DTYPE(Na_i**3)*DTYPE(Ca_o)
                              -jnp.exp(DTYPE((gamma_-1.)*F/R/T)*V)*Na_o3*conCa) \
           /(denom_NaCa+DTYPE(1e-10))

    i_bNa = DTYPE(g_bNa)*(V-E_Na)
    carow_3 = DTYPE(maxI_pCa)*conCa/(DTYPE(0.0005)+conCa) \
              - DTYPE(g_bCa*R*T/2./F)*jnp.log(DTYPE(Ca_o)/(conCa+DTYPE(1e-10)))
    i_pCa = carow_3 + V*DTYPE(g_bCa)

    # SR
    i_rel = u**2*vv*DTYPE(k_rel)*w*(Ca_rel-conCa)
    fn    = DTYPE(C_Fn1)*i_rel - DTYPE(C_Fn2)*(i_CaL-DTYPE(0.4)*i_NaCa)
    # logistic functions of the SR release flux written with jax.nn.sigmoid: identical values, but the forward-mode
    # tangent stays finite when fn is large and negative (exp(...) overflows and 1/(1+exp) gave inf/inf = NaN)
    u_inf    = jax.nn.sigmoid((fn-DTYPE(3.4175e-13))/DTYPE(13.67e-16*TRIG_W))
    tau_u    = DTYPE(8.)
    v_inf    = jax.nn.sigmoid((DTYPE(6.835e-14)-fn)/DTYPE(13.67e-16*TRIG_W))
    tau_v_sr = DTYPE(1.91)+DTYPE(2.09)*jax.nn.sigmoid((fn-DTYPE(3.4175e-13))/DTYPE(13.67e-16*TRIG_W))

    carow_1 = DTYPE(maxI_up)/(DTYPE(1.)+DTYPE(K_up)/(conCa+DTYPE(1e-10)))
    di_ups  = carow_1 - DTYPE(C_Iupl)*Ca_up
    i_tr    = (Ca_up-Ca_rel)/DTYPE(tau_tr)

    I_ion = i_Na+i_K1+i_to+i_Kur+i_Kr+i_Ks+i_CaL+i_pCa+i_NaCa+i_bNa+i_NaK

    # Ca buffering — clamp carow_2 to prevent division by zero
    carow_2 = jnp.clip(
        (DTYPE(maxTrpn*KmTrpn)/(conCa+DTYPE(KmTrpn))**2
         +DTYPE(maxCmdn*KmCmdn)/(conCa+DTYPE(KmCmdn))**2+DTYPE(1.))/DTYPE(C_B1c)/DTYPE(1000.),
        DTYPE(1e-10), None)

    diff_Ca_i   = jnp.clip(
        (DTYPE(C_B1d)*(DTYPE(2.)*i_NaCa-i_pCa-i_CaL)
         -DTYPE(C_B1e)*di_ups+i_rel)/carow_2,
        DTYPE(-1e4), DTYPE(1e4))
    diff_Ca_up  = di_ups - i_tr*DTYPE(C_dCaup)
    diff_Ca_rel = (i_tr-i_rel)/(DTYPE(1.)+DTYPE(C_dCa_rel)/(Ca_rel+DTYPE(KmCsqn))**2)

    # Rush-Larsen gate updates
    m_new    = rl(m,    m_inf,    tau_m)
    h_new    = rl(h,    h_inf,    tau_h)
    j_new    = rl(j,    j_inf,    tau_j)
    oa_new   = rl(oa,   oa_inf,   tau_oa)
    oi_new   = rl(oi,   oi_inf,   tau_oi)
    ua_new   = rl(ua,   ua_inf,   tau_ua)
    ui_new   = rl(ui,   ui_inf,   tau_ui)
    xr_new   = rl(xr,   xr_inf,   tau_xr)
    xs_new   = rl(xs,   xs_inf,   tau_xs)
    d_new    = rl(d,    d_inf,    tau_d)
    f_new    = rl(f,    f_inf,    tau_f)
    f_Ca_new = rl(f_Ca, f_Ca_inf, tau_fCa)
    u_new    = rl(u,    u_inf,    tau_u)
    vv_new   = rl(vv,   v_inf,    tau_v_sr)
    w_new    = rl(w,    w_inf,    tau_w)

    # Euler for V: dV/dt = -(I_ion - I_stim)/C_m, C_m=100pF normalised to 1
    # I_ion in pA/pF → dV in mV/ms when C_m=1
    V_new      = jnp.clip(V - dt*(I_ion - I_stim_vec), DTYPE(-150.), DTYPE(60.))
    Ca_i_new   = jnp.clip(Ca_i   + dt*diff_Ca_i,   DTYPE(1e-4), DTYPE(1e4))
    Ca_up_new  = jnp.clip(Ca_up  + dt*diff_Ca_up,  DTYPE(0.),   DTYPE(30.))
    Ca_rel_new = jnp.clip(Ca_rel + dt*diff_Ca_rel, DTYPE(0.),   DTYPE(30.))

    return (V_new,m_new,h_new,j_new,oa_new,oi_new,ua_new,ui_new,
            xr_new,xs_new,d_new,f_new,f_Ca_new,u_new,vv_new,w_new,
            Ca_i_new,Ca_up_new,Ca_rel_new)


def conductances(p6):
    """Map fitted multipliers -> CRN conductances (g_Ks fixed)."""
    return (DTYPE(NOM['g_Na'])*p6[0], DTYPE(NOM['g_to'])*p6[1], DTYPE(NOM['g_Kr'])*p6[2],
            DTYPE(NOM['g_CaL'])*p6[3], DTYPE(NOM['g_K1'])*p6[4], DTYPE(NOM['g_Ks']))

# ══════════════════════════════════════════════════════════════════════════
# [2] FEATURE EXTRACTORS (identical to the mMS GT script)
# ══════════════════════════════════════════════════════════════════════════
def wyatt_ari(egm, i_start, dt, win=4000):
    seg = egm[i_start:i_start+win]
    if len(seg) < 10 or np.ptp(seg) < 1e-9: return np.nan
    d1 = np.gradient(seg, dt); grd = max(1, int(50./dt))
    i_at = int(np.argmin(d1))
    cap = min(len(d1), i_at + int(400./dt))
    if i_at + grd >= cap: return np.nan
    i_rt = i_at + grd + int(np.argmax(d1[i_at+grd:cap]))
    return float((i_rt - i_at)*dt)

def get_at(egm, i_start, dt, win=200):
    seg = egm[i_start:i_start+win]
    if len(seg) < 5: return np.nan
    return float(np.argmin(np.gradient(seg, dt))*dt)

def get_slew(egm, i_start, dt, win=200):
    seg = egm[i_start:i_start+win]
    if len(seg) < 5: return np.nan
    return float(np.max(np.abs(np.gradient(seg, dt))))

def get_p2p(egm, i_start, dt, win=4000):
    seg = egm[i_start:i_start+win]
    if len(seg) < 5: return np.nan
    return float(np.ptp(seg))

# ══════════════════════════════════════════════════════════════════════════
# [3] MESH + PATCH BUILDER  (same geometry code as the mMS GT script)
# ══════════════════════════════════════════════════════════════════════════
def build_patch(case):
    # the patch (nodes, triangles, HD grid, fibres, pacing edge) comes from patch_geometry.npz, which holds the
    # same patch the original runs cut from the clinical mesh (2,894 nodes, LEFT pacing edge, 499 strip nodes)
    _pg = np.load(PATCH_GEOMETRY_PATH, allow_pickle=True)
    lv = np.array(_pg["lv"], dtype=np.float64); le = np.array(_pg["le"], dtype=np.int64)
    HD16 = np.array(_pg["HD16"], dtype=np.float64); fib3 = np.array(_pg["fib3"], dtype=np.float64)
    cs_edge = str(_pg["cs_edge"][0]); Np = len(lv)
    vc = lv*1e-4
    print(f"  [{case}] {Np:,} nodes  pacing [{cs_edge}]:", end=' ')

    nn = np.zeros((Np, 3)); ar = np.zeros(Np)
    for (i, j, k) in le:
        cr = np.cross(lv[j]-lv[i], lv[k]-lv[i]); a = 0.5*np.linalg.norm(cr)
        n_ = cr/(np.linalg.norm(cr)+1e-12)
        for nd in (i, j, k): ar[nd] += a/3.; nn[nd] += a*n_
    for i in range(Np):
        n = np.linalg.norm(nn[i])
        if n > 1e-12: nn[i] /= n
    m_j = jnp.array(1./(CM*np.maximum(ar*1e-8, max(1e-8, np.percentile((ar*1e-8)[ar > 0], 5)))))

    ec = defaultdict(float)
    for (i, j, k) in le:
        pi, pj, pk = vc[i], vc[j], vc[k]
        for (u, v_, w, a_, b_) in [(i, j, k, pj-pi, pk-pi), (j, i, k, pi-pj, pk-pj),
                                   (k, i, j, pi-pk, pj-pk)]:
            c = np.linalg.norm(np.cross(a_, b_))
            if c < 1e-14: continue
            ec[(min(v_, w), max(v_, w))] += 0.5*np.dot(a_, b_)/c
    eu = np.array([u for u, v_ in ec.keys()], dtype=np.int32)
    ev = np.array([v_ for u, v_ in ec.keys()], dtype=np.int32)
    ecot = np.array(list(ec.values()), dtype=np.float64)
    ed = vc[ev]-vc[eu]; ed /= np.linalg.norm(ed, axis=1, keepdims=True)+1e-12
    en = (nn[eu]+nn[ev])/2.; en /= np.linalg.norm(en, axis=1, keepdims=True)+1e-12
    eu_j = jnp.array(eu); ev_j = jnp.array(ev)
    ecot_j = jnp.array(ecot); ed_j = jnp.array(ed); en_j = jnp.array(en)
    fib3_j = jnp.array(np.tile(fib3, (len(eu), 1)) if fib3.ndim == 1 else fib3)

    ep = HD16[EDGE_ELEC[cs_edge]]; ec2 = ep.mean(0)
    c5c = HD16[CLIQUES[CENTRE_CLIQUE]].mean(0)
    wf = (c5c-ec2)/(np.linalg.norm(c5c-ec2)+1e-12); lc = ec2-wf*OFFSET_MM*1000.
    _, _, Ve = np.linalg.svd(ep-ec2, full_matrices=False); ld = Ve[0]
    projs = (ep-ec2)@ld; hl = max(np.ptp(projs)/2., 1.)+2000.
    d = lv-lc; pal = d@ld; perp = d-np.outer(pal, ld)
    dist = np.linalg.norm(perp, axis=1)
    mk = (dist <= STRIP_PAD_MM*1000.) & (np.abs(pal) <= hl)
    if mk.sum() < 10: mk = (dist <= STRIP_PAD_MM*2000.) & (np.abs(pal) <= hl+2000.)
    mk_j = jnp.array(mk.astype(np.float64))
    print(f"{int(mk.sum())} nodes")

    sig_r = 1
    W_np = np.zeros((16, Np))
    for e_ in range(16):
        r_ = np.maximum(np.linalg.norm(vc-HD16[e_]*1e-4, axis=1), 1e-6)
        W_np[e_] = (1./r_)*(sig_r/(4.*np.pi))
    W_j = jnp.array(W_np)

    NT = int(TOTAL_MS/DT_SIM); t_ms = np.arange(NT)*DT_SIM
    SV = np.zeros((len(PROTS), NT), dtype=bool)            # stimulus schedule of every sequence
    for _j, _P in enumerate(PROTS):
        for _ts in _P['stim']:
            SV[_j] |= (t_ms >= _ts) & (t_ms < _ts+STIM_DUR)
    SV_ALL = jnp.array(SV)

    dth = 0.0
    def _build_w(g_il):
        g_it = g_il/4.
        fs = fib3_j-jnp.sum(fib3_j*en_j, axis=1, keepdims=True)*en_j
        fs /= jnp.linalg.norm(fs, axis=1, keepdims=True)+1e-12
        pp = jnp.cross(en_j, fs); pp /= jnp.linalg.norm(pp, axis=1, keepdims=True)+1e-12
        fr = fs*jnp.cos(dth)+pp*jnp.sin(dth)
        fr /= jnp.linalg.norm(fr, axis=1, keepdims=True)+1e-12
        cos2 = jnp.sum(ed_j*fr, axis=1)**2
        return jnp.abs(ecot_j)*((g_il/(BETA*CM))*cos2+(g_it/(BETA*CM))*(1.-cos2))

    def make_run(n_ion=N_ION):
        @jax.jit
        def run_3d(p6, svq):
            """p6 = [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL] -> phi (NT, 16)."""
            gNa, gto, gKr, gCaL, gK1, gKs = conductances(p6)
            w = _build_w(p6[5])
            dt_sub = DTYPE(DT_SIM/(2*n_ion)); alpha = DTYPE(DT_SIM/2.)
            def spmv(x):
                Kx = jnp.zeros(Np, dtype=x.dtype)
                Kx = Kx.at[eu_j].add(-w*x[ev_j]+w*x[eu_j])
                Kx = Kx.at[ev_j].add(-w*x[eu_j]+w*x[ev_j])
                return Kx
            def _cn(Vn):
                rhs = Vn-alpha*m_j*spmv(Vn)
                out, _ = jax.scipy.sparse.linalg.cg(
                    lambda x: x+alpha*m_j*spmv(x), rhs, x0=Vn, tol=CG_TOL, maxiter=CG_MAXITER)
                return out
            to_norm = lambda Vm: (Vm-DTYPE(V_SHIFT))/DTYPE(V_SCALE)
            to_mv   = lambda Vn: Vn*DTYPE(V_SCALE)+DTYPE(V_SHIFT)
            def ion_half(state, I_s):
                return jax.lax.fori_loop(
                    0, n_ion,
                    lambda _, s: crn_ion_step(s, I_s, dt_sub, gNa, gto, gKr, gCaL, gK1, gKs),
                    state)
            def scan_fn(state, sv_):
                I_s = jnp.where(sv_, DTYPE(STIM_AMP)*mk_j, jnp.zeros(Np, dtype=DTYPE))
                state = ion_half(state, I_s)
                Vn = _cn(to_norm(state[0]))
                state = (to_mv(Vn),) + state[1:]
                state = ion_half(state, I_s)
                return state, W_j @ to_norm(state[0])
            _, phi_T = jax.lax.scan(jax.checkpoint(scan_fn), crn_initial(Np), svq)
            return phi_T
        return run_3d

    def get_egm(phi):
        egms = []
        for ci in range(9):
            c_ = CLIQUES[ci]
            egms.append((phi[:, c_[0]]-phi[:, c_[1]]-phi[:, c_[2]]+phi[:, c_[3]])/4.)
        return np.array(egms)

    return make_run(), get_egm, DT_SIM, SV_ALL, NT

# ══════════════════════════════════════════════════════════════════════════
# [4] 0-D SANITY CHECK of the ionic model at the chosen parameters
# ══════════════════════════════════════════════════════════════════════════
def check_0d(p6, label, bcl=500., nbeats=3, n_ion=N_ION):
    dt = DT_SIM/(2*n_ion)
    n = int(nbeats*bcl/dt); t = np.arange(n)*dt
    sv = np.zeros(n, bool)
    for k in range(nbeats): sv |= (t >= 10+k*bcl) & (t < 10+k*bcl+STIM_DUR)
    g = conductances(jnp.asarray(p6, dtype=DTYPE))
    @jax.jit
    def go(sv_j):
        def f(s, x):
            s = crn_ion_step(s, jnp.where(x, DTYPE(STIM_AMP), DTYPE(0.))*jnp.ones(1),
                             DTYPE(dt), *g)
            return s, s[0][0]
        return jax.lax.scan(f, crn_initial(1), sv_j)[1]
    V = np.array(go(jnp.array(sv)))
    i0 = int((nbeats-1)*bcl/dt); v = V[i0:]
    vrest = v[:int(8/dt)].mean(); vpk = v.max(); ipk = v.argmax()
    thr = vpk-0.9*(vpk-vrest); idx = np.where(v[ipk:] < thr)[0]
    apd90 = idx[0]*dt if len(idx) else float('nan')
    ok = (vpk > 0) and (150. < apd90 < 400.)
    print(f"  0-D [{label}] last beat: Vrest={vrest:.1f}  Vpeak={vpk:.1f} mV  "
          f"dVmax={np.max(np.diff(v))/dt:.0f} V/s  APD90={apd90:.0f} ms  "
          f"{'OK' if ok else 'WARNING: check parameters'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# [5] INVERSION  -  same pipeline as the mMS fit (GT-free)
#     Phase 1: Nelder-Mead, 9-clique S2 MSE (all 6 free)
#     Phase 2: L-BFGS-B with exact jacfwd gradient, 9-clique S1 + S2 MSE (warm start from Phase 1)
#     Loss = mean_c[MSE_S2/p2p_S2^2] + 0.5*mean_c[MSE_S1/p2p_S1^2]   (identical to the mMS fit)
# ══════════════════════════════════════════════════════════════════════════
from scipy.optimize import minimize

GT_DIR   = os.environ.get("GT_DIR", ".")              # folder holding egm_pipeline_for_jax_{case}/gt_crn_egm_{case}.npz
GT_FILE  = os.environ.get("GT_FILE", "gt_crn_egm_{case}.npz")
WORK_DIR = os.environ.get("WORK_DIR", "./crn_results")
os.makedirs(WORK_DIR, exist_ok=True)

# bounds (logit-parametrised, as in the mMS fit)   [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL]
# Physiological range per parameter, shrunk inward by 20 % on each side (same construction as the mMS fit);
# the optimiser starts from the midpoint of the shrunk range (INIT_FRAC = 0.5).
LB_RAW6 = jnp.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.05], dtype=DTYPE)
UB_RAW6 = jnp.array([1.6, 1.6, 1.6, 1.6, 1.6, 0.40], dtype=DTYPE)
_pad6 = (UB_RAW6 - LB_RAW6) * 0.20
LB6 = LB_RAW6 + _pad6
UB6 = UB_RAW6 - _pad6
def get_p6(lp): return LB6 + (UB6-LB6)*jax.nn.sigmoid(lp)
def inv_p6(p):
    ps = jnp.clip(jnp.asarray(p, dtype=DTYPE), LB6+1e-6, UB6-1e-6)
    return jax.scipy.special.logit((ps-LB6)/(UB6-LB6))

INIT_FRAC      = float(os.environ.get("INIT_FRAC", 0.5))
LOSS_TARGET_P1 = float(os.environ.get("LOSS_TARGET_P1", 1e-5))   # CRN losses start ~1e-3, so 1e-4 is far too loose
MAX_EVALS_P1   = int(os.environ.get("MAX_EVALS_P1", 300))
PATIENCE       = int(os.environ.get("PATIENCE", 40))              # Phase 1 stops after this many evaluations without a RELATIVE improvement of P1_REL_TOL
P1_REL_TOL     = float(os.environ.get("P1_REL_TOL", 0.01))        # 1 % of the best loss so far
P1_MIN_EVALS   = int(os.environ.get("P1_MIN_EVALS", 60))          # the patience rule cannot stop Nelder-Mead before this many evaluations
NM_STEP        = 1.5
NM_RESTARTS    = int(os.environ.get("NM_RESTARTS", 8))           # number of Nelder-Mead restarts after the first run
NM_STEP_DECAY  = float(os.environ.get("NM_STEP_DECAY", 0.4))     # simplex size of every restart = previous size x this factor
NM_RESTART_GAIN= float(os.environ.get("NM_RESTART_GAIN", 0.05))  # stop restarting when a round improves the loss by less than this fraction
LOSS_STOP_P2   = 1e-7; GRAD_STOP_P2 = 1e-6
PLATEAU_WIN_P2 = 5;    PLATEAU_TOL_P2 = 1e-4
MAX_EVALS_P2   = int(os.environ.get("MAX_EVALS_P2", 60))
GRAD_MODE      = os.environ.get("GRAD_MODE", "jacfwd").lower()   # paper: jacfwd;  # 'auto' | 'jacfwd' | 'fd'.  auto: forward-mode gradient, CHECKED against central finite differences at
                                                                #   every evaluation; the first time they disagree it switches to FD for the rest of the run.
GRAD_CHECK_RTOL = float(os.environ.get("GRAD_CHECK_RTOL", 0.10))   # relative tolerance on ||g_ad - g_fd|| / ||g_fd||
GRAD_CHECK_ATOL = float(os.environ.get("GRAD_CHECK_ATOL", 1e-5))   # absolute floor (FD noise level) so tiny gradients near the optimum do not trigger a switch
FD_H           = float(os.environ.get("FD_H", 3e-4))
P1_MODE        = os.environ.get("P1_MODE", "save").lower()     # 'load' re-uses the Phase-1 cache
SKIP_P1        = os.environ.get("SKIP_P1", "0") == "1"          # 1: no Nelder-Mead, L-BFGS-B starts directly at the midpoint of the bounds

# ---- which parameters are recovered (the others are held at their nominal value) -------------
SHORT   = ['gNa', 'gto', 'gKr', 'gCaL', 'gK1', 'G_IL']
LABELS  = ['g_Na', 'g_to', 'g_Kr', 'g_CaL', 'g_K1', 'G_il']          # names used in the printed output
FREE_NAMES = [n.strip() for n in os.environ.get("FREE", "gNa,gto,gKr,gCaL,gK1,G_IL").split(",")]
FREE_IDX = np.array([SHORT.index(n) for n in FREE_NAMES]); NFREE = len(FREE_IDX)
FIXED_VAL = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.225])             # nominal value of any parameter that is not recovered
FREE_TAG = "_f" + "".join(str(i) for i in FREE_IDX)

def fmt_free(p6):
    return "  ".join(f"{LABELS[i]}={p6[i]:.3f}" if i != 5 else f"{LABELS[i]}={p6[i]:.4f}" for i in FREE_IDX)

# clique EGM operator: egm (NT,9) = phi (NT,16) @ CQ (16,9)
CQ = np.zeros((16, 9))
for _ci, _c in enumerate(CLIQUES):
    CQ[_c[0], _ci] += .25; CQ[_c[1], _ci] -= .25; CQ[_c[2], _ci] -= .25; CQ[_c[3], _ci] += .25
CQ_j = jnp.array(CQ)

def invert(case):
    t_case = time.time()
    print(f"\n{'='*60}\nCASE: {case}\n{'='*60}")
    gt_path = os.path.join(GT_DIR, f"egm_pipeline_for_jax_{case}", GT_FILE.format(case=case))
    gt = np.load(gt_path, allow_pickle=True)
    n_ci = len(PROTS)
    assert [float(x) for x in gt['ci_ms']] == CI_LIST, "coupling intervals of the GT file differ from CI_LIST"
    print(f"  Target file: {gt_path}\n  {n_ci} sequences (S1, S1, S2), S2 coupling intervals {CI_LIST} ms")
    # ---- normalisation of every beat window ---------------------------------------------------------------
    # NORM_SPLIT_MS > 0 (default 60): each beat is split into an ACTIVATION window (first NORM_SPLIT_MS after the stimulus) and a
    # REPOLARISATION window (the rest); each is normalised by its OWN per-clique peak-to-peak and the two count equally.
    # This stops the large activation spike from scaling the much smaller T wave out of the loss.  NORM_SPLIT_MS=0: old behaviour
    # (one peak-to-peak per beat).
    NORM_SPLIT = int(float(os.environ.get("NORM_SPLIT_MS", 60.))/DT_SIM)
    def _norm_wt(seg):                                              # seg (9, w) -> NRM (9, w), WT (w,) with sum(WT) = 1
        w = seg.shape[1]
        if NORM_SPLIT <= 0 or w <= NORM_SPLIT + 10:
            p = np.ptp(seg, axis=1)**2 + 1e-18
            return np.repeat(p[:, None], w, axis=1), np.full(w, 1./w)
        pa = np.ptp(seg[:, :NORM_SPLIT], axis=1)**2 + 1e-18; pr = np.ptp(seg[:, NORM_SPLIT:], axis=1)**2 + 1e-18
        nrm = np.concatenate([np.repeat(pa[:, None], NORM_SPLIT, axis=1), np.repeat(pr[:, None], w-NORM_SPLIT, axis=1)], axis=1)
        wt = np.concatenate([np.full(NORM_SPLIT, 0.5/NORM_SPLIT), np.full(w-NORM_SPLIT, 0.5/(w-NORM_SPLIT))])
        return nrm, wt
    targets = []                       # per sequence: [(start index, width, EGM window (9,w), NRM (9,w), WT (w,)) for every beat]
    for _j, _P in enumerate(PROTS):
        _b = []
        for _k in range(len(_P['stim'])):
            _sg = np.asarray(gt[f"segs_ci{_j}_b{_k}_gt"]); _nr, _wt = _norm_wt(_sg)
            _b.append((int(_P['beat_start'][_k]/DT_SIM), _sg.shape[1], jnp.array(_sg), jnp.array(_nr), jnp.array(_wt)))
        targets.append(_b)
    print(f"  Normalisation: " + (f"activation (first {NORM_SPLIT*DT_SIM:.0f} ms) and repolarisation windows separately" if NORM_SPLIT > 0 else "one peak-to-peak per beat"))
    n_cond = len(PROTS[0]['stim']) - 1
    print(f"  Per sequence: {n_cond} conditioning beats + premature beat; windows (ms) {[b[1]*DT_SIM for b in targets[0]]}")

    run_3d, get_egm, dt_sim, SV_ALL, NT = build_patch(case)
    x_fix = inv_p6(FIXED_VAL)                                         # logit of the nominal (held) values
    def expand(z): return x_fix.at[FREE_IDX].set(z)                   # reduced vector -> full 6-vector
    def P6(z): return get_p6(expand(jnp.asarray(z, dtype=DTYPE)))     # reduced logit -> physical parameters
    def sim_all(p6): return jax.vmap(lambda sv_: run_3d(p6, sv_))(SV_ALL)       # (n_ci, NT, 16): all sequences in one batch

    # ---- losses (JAX; used by both phases) ------------------------------------------------
    # Loss = mean over sequences of [ premature-beat MSE + w_s1 * mean over conditioning beats of MSE ]  (per-clique normalised)
    def loss_jax(lp6, w_s1=0.5):
        phis = sim_all(get_p6(lp6)); total = 0.
        for j in range(n_ci):
            i_b, w_b, seg_b, nrm_b, wt_b = targets[j][-1]
            l2 = jnp.mean(jnp.sum(((phis[j, i_b:i_b+w_b] @ CQ_j).T-seg_b)**2*wt_b[None, :]/nrm_b, axis=1))
            if w_s1 == 0.: total = total + l2; continue
            l1 = 0.
            for (i_c, w_c, seg_c, nrm_c, wt_c) in targets[j][:-1]:
                l1 = l1 + jnp.mean(jnp.sum(((phis[j, i_c:i_c+w_c] @ CQ_j).T-seg_c)**2*wt_c[None, :]/nrm_c, axis=1))
            total = total + l2 + w_s1*l1/n_cond
        return total/n_ci
    def loss_red(z, w_s1=0.5): return loss_jax(expand(z), w_s1)
    loss_p1_jit = jax.jit(lambda z: loss_red(z, 0.))     # Phase 1: S2 MSE only
    loss_p2_jit = jax.jit(loss_red)
    grad_p2_jit = jax.jit(jax.jacfwd(loss_red))           # exact forward-mode gradient (NFREE tangents)
    loss_p1_batch = jax.jit(jax.vmap(lambda z: loss_red(z, 0.)))
    loss_p2_batch = jax.jit(jax.vmap(loss_red))
    def fd_value_and_grad(x):                         # central FD in logit space; all 12 perturbed runs in ONE batched call
        E = FD_H*np.eye(NFREE); pts = np.vstack([x[None], x+E, x-E])
        Lb = np.array(loss_p2_batch(jnp.array(pts, dtype=DTYPE)))
        # sporadic forward-run failures give a loss 10-1e4 x too large (same parameters, different result between runs); the true
        # change over the FD step is ~1e-6, so any run above 1.5 x the median is a failure: re-run the batch and keep the smaller value
        for _try in range(4):
            bad = ~np.isfinite(Lb) | (Lb > 1.5*np.nanmedian(Lb))
            if not bad.any(): break
            Lb2 = np.array(loss_p2_batch(jnp.array(pts, dtype=DTYPE)))
            print(f"    (spurious loss spike in {int(bad.sum())} of {len(Lb)} runs, re-evaluated)", flush=True)
            Lb = np.where(bad, np.minimum(np.nan_to_num(Lb, nan=1e9), np.nan_to_num(Lb2, nan=1e9)), Lb)
        return float(Lb[0]), (Lb[1:1+NFREE]-Lb[1+NFREE:1+2*NFREE])/(2*FD_H)

    # ---- Phase 1 ---------------------------------------------------------------------------
    print(f"\n  {'─'*56}\n  PHASE 1 - Nelder-Mead, S2 MSE ({NFREE} free: {', '.join(LABELS[i] for i in FREE_IDX)})\n  {'─'*56}")
    t_p1 = time.time()
    x_best = np.array(inv_p6(LB6 + INIT_FRAC*(UB6-LB6)), dtype=np.float64)[FREE_IDX]
    p1_file = os.path.join(WORK_DIR, f"p1_cache_crn_{case}{FREE_TAG}{SMOOTH_TAG}.npz")
    converged = False; best_loss = [float('nan')]
    if P1_MODE == 'load' and os.path.exists(p1_file):
        x_best = np.load(p1_file)['x_logit'].copy(); converged = True
        best_loss = [float(loss_p1_jit(jnp.array(x_best)))]
        print(f"  Phase 1 LOADED from cache: {p1_file}")
    p_init = np.array(P6(x_best))
    print("  Start (midpoint of the bounds): " + fmt_free(p_init))
    print("  Held fixed at nominal:          " + ("  ".join(f"{LABELS[i]}={FIXED_VAL[i]:.3f}" for i in range(6) if i not in FREE_IDX) or "none"))

    if SKIP_P1 and not converged:
        converged = True; best_loss = [float(loss_p1_jit(jnp.array(x_best, dtype=DTYPE)))]
        print("  SKIP_P1=1: Nelder-Mead skipped, L-BFGS-B starts at the midpoint of the bounds")
    if not converged:
        def make_simplex(x0_, step):
            sx = np.tile(x0_, (NFREE+1, 1))
            for i in range(NFREE): sx[i+1, i] += step
            return sx
        simplex = make_simplex(x_best, NM_STEP)
        l_simplex = np.array(loss_p1_batch(jnp.array(simplex)))     # whole initial simplex in one batched run
        print(f"  Initial simplex losses: {np.round(l_simplex, 5)}")
        n_eval = [0]; n_round = [0]; best_x = [simplex[int(np.argmin(l_simplex))].copy()]
        best_loss = [float(l_simplex.min())]; plateau = [0]; t0 = time.time()
        def f_nm(x):
            v = float(loss_p1_jit(jnp.array(x, dtype=DTYPE)))
            if (not np.isfinite(v)) or v > 50.*best_loss[0]:                     # possible spurious failure: evaluate again, keep the smaller
                v2 = float(loss_p1_jit(jnp.array(x, dtype=DTYPE)))
                if v2 < v: print(f"    (loss {v:.4g} -> {v2:.4g} on re-evaluation: spurious spike)", flush=True); v = v2
            n_eval[0] += 1; n_round[0] += 1
            if v < best_loss[0]*(1.-P1_REL_TOL): plateau[0] = 0          # a real (>= P1_REL_TOL) improvement resets the patience counter
            else: plateau[0] += 1
            if v < best_loss[0]: best_loss[0] = v; best_x[0] = x.copy()   # always keep the best point
            p_ = np.array(P6(x))
            print(f"  e={n_eval[0]:3d} loss={v:.5f}  {fmt_free(p_)}  t={time.time()-t0:.0f}s", flush=True)
            if v <= LOSS_TARGET_P1: raise StopIteration(f"loss={v:.6f} <= {LOSS_TARGET_P1}")
            if plateau[0] >= PATIENCE and n_round[0] >= P1_MIN_EVALS: raise StopIteration(f"plateau {PATIENCE} evals")
            return v
        # Nelder-Mead with RESTARTS: a simplex that has collapsed in a narrow valley is rebuilt around the best point with a smaller step
        # and run again; the restarts stop when a round no longer improves the loss by NM_RESTART_GAIN (relative) or the loss target is met.
        step = NM_STEP
        for rnd in range(NM_RESTARTS+1):
            if rnd > 0:
                step *= NM_STEP_DECAY; simplex = make_simplex(best_x[0], step)
                l_simplex = np.array(loss_p1_batch(jnp.array(simplex)))
                if float(l_simplex.min()) < best_loss[0]: best_loss[0] = float(l_simplex.min()); best_x[0] = simplex[int(np.argmin(l_simplex))].copy()
                print(f"\n  >>> Nelder-Mead RESTART {rnd}/{NM_RESTARTS}: new simplex around the best point, step {step:.3g}, best loss so far {best_loss[0]:.4g}", flush=True)
            loss_round_start = best_loss[0]; plateau[0] = 0; n_round[0] = 0; target_hit = False
            try:
                minimize(f_nm, best_x[0], method='Nelder-Mead',
                         options={'maxiter': 2000, 'maxfev': MAX_EVALS_P1, 'xatol': 1e-5, 'fatol': 1e-9,
                                  'initial_simplex': simplex, 'disp': False})
            except StopIteration as e:
                print(f"\n  -> {e}")
                if best_loss[0] <= LOSS_TARGET_P1: target_hit = True
            if target_hit: break
            gain = 1. - best_loss[0]/max(loss_round_start, 1e-300)
            print(f"  round {rnd}: loss {loss_round_start:.4g} -> {best_loss[0]:.4g}  (gain {100*gain:.1f} %)", flush=True)
            if rnd > 0 and gain < NM_RESTART_GAIN: print("  -> restart no longer helps, Phase 1 ends"); break
        x_best = best_x[0]
        np.savez(p1_file, x_logit=x_best); print(f"  Phase 1 cache saved: {p1_file}")
    p_p1 = np.array(P6(x_best))
    print(f"  Phase 1 done: loss={best_loss[0]:.6f}  time={(time.time()-t_p1)/60.:.1f} min")

    # ---- Phase 2 ---------------------------------------------------------------------------
    print(f"\n  {'─'*58}\n  PHASE 2 - L-BFGS-B ({GRAD_MODE} gradient), S1+S2 MSE ({NFREE} free)\n  {'─'*58}")
    t_p2 = time.time(); x0 = x_best.copy()
    gmode = ['fd' if GRAD_MODE == 'fd' else 'jacfwd']                 # gradient actually in use (auto may switch jacfwd -> fd)
    if GRAD_MODE == 'auto': print(f"  Gradient check ON: jacfwd vs central FD (h={FD_H}) at every evaluation; tolerance {GRAD_CHECK_RTOL:.0%} relative + {GRAD_CHECK_ATOL:g} absolute. "
                                  "Switch to FD for good at the first disagreement.")
    print("  Warm start: " + fmt_free(p_p1))
    n_p2 = [0]; best_l2 = [1e9]; best_x2 = [x0.copy()]; hist = []; t_grad = [0.]
    def f_p2(x):
        xj = jnp.array(x, dtype=DTYPE); p6 = np.array(get_p6(expand(xj))); tg = time.time()
        if gmode[0] == 'fd': v, g = fd_value_and_grad(np.asarray(x, dtype=np.float64))
        else:
            v = float(loss_p2_jit(xj)); g = np.array(grad_p2_jit(xj))
            if GRAD_MODE == 'auto':                                   # check the forward-mode gradient against finite differences
                v_fd, g_fd = fd_value_and_grad(np.asarray(x, dtype=np.float64))
                dn = float(np.linalg.norm(g-g_fd)); fn = float(np.linalg.norm(g_fd))
                ok = bool(np.all(np.isfinite(g)) and np.isfinite(dn) and dn <= GRAD_CHECK_ATOL + GRAD_CHECK_RTOL*fn)
                print(f"    grad check: |g_ad|={float(np.linalg.norm(g)):.3e}  |g_fd|={fn:.3e}  |g_ad-g_fd|/|g_fd|={dn/(fn+1e-30):.2e}  -> "
                      + ("OK" if ok else "MISMATCH: switching to finite differences for the rest of the run"), flush=True)
                if not ok: gmode[0] = 'fd'; v, g = v_fd, g_fd
        t_grad[0] += time.time()-tg; gn = float(np.linalg.norm(g)); n_p2[0] += 1
        if v < best_l2[0]-1e-12: best_l2[0] = v; best_x2[0] = x.copy()
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            print(f"  e={n_p2[0]:2d} non-finite loss/gradient (loss={v}, |g|={gn}); stopping Phase 2 at the best point so far", flush=True)
            raise StopIteration
        hist.append(v)
        print(f"  e={n_p2[0]:2d} loss={v:.3e}  {fmt_free(p6)}  |grad|={gn:.2e}  t={time.time()-t_p2:.0f}s", flush=True)
        if v <= LOSS_STOP_P2 and gn <= GRAD_STOP_P2:
            print(f"  Converged: loss={v:.2e} |g|={gn:.2e}"); raise StopIteration
        if len(hist) >= PLATEAU_WIN_P2:
            rec = hist[-PLATEAU_WIN_P2:]; rel = (rec[0]-rec[-1])/(rec[0]+1e-12)
            if rel < PLATEAU_TOL_P2 and gn < 1e-6:
                print(f"  Plateau: rel_imp={rel:.2e}"); raise StopIteration
        return v, g
    try:
        minimize(f_p2, x0, method='L-BFGS-B', jac=True,
                 options={'maxiter': MAX_EVALS_P2, 'maxfun': MAX_EVALS_P2*11, 'ftol': 1e-12, 'gtol': 1e-6,
                          'maxcor': 20})
    except StopIteration:
        pass
    xf = best_x2[0].copy(); p_p2 = np.array(P6(xf))
    print(f"  Phase 2 done: loss={best_l2[0]:.2e}  evals={n_p2[0]}  time={(time.time()-t_p2)/60.:.1f} min")
    print("  Recovered: " + fmt_free(p_p2) + f"   (gradient used at the end: {gmode[0]})")
    print(f"  PHASE2_SUMMARY case={case} method={GRAD_MODE}->{gmode[0]} loss={best_l2[0]:.3e} evals={n_p2[0]} "
          f"phase2_s={time.time()-t_p2:.1f} grad_s={t_grad[0]:.1f} params={' '.join(f'{v:.5f}' for v in p_p2)}", flush=True)
    np.savez(os.path.join(WORK_DIR, f"results_crn_{case}_{GRAD_MODE}{FREE_TAG}{SMOOTH_TAG}.npz"), p_phase1=p_p1, p_phase2=p_p2,
             loss_p1=np.array([best_loss[0]]), loss_p2=np.array([best_l2[0]]), n_evals_p2=np.array([n_p2[0]]),
             p2_time_s=np.array([time.time()-t_p2]), x_logit_p2=xf, case=np.array([case]))
    # ---- recovered vs GT EGMs (one figure per coupling interval) ---------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        phis = np.array(sim_all(jnp.array(p_p2, dtype=DTYPE)))
        for j, P in enumerate(PROTS):
            eg = phis[j] @ CQ
            gt_all = np.concatenate([np.asarray(gt[f"segs_ci{j}_b{k}_gt"]) for k in range(len(P['stim']))], axis=1)
            i0 = int(P['beat_start'][0]/DT_SIM); n_all = gt_all.shape[1]
            t_all = (i0 + np.arange(n_all))*dt_sim
            fig, axs = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
            for ci, ax in enumerate(axs.ravel()):
                ax.plot(t_all, gt_all[ci], 'k--', lw=1.2, label='GT (CRN)')
                ax.plot(t_all, eg[i0:i0+n_all, ci], color='#0072B2', lw=0.9, label='Recovered')
                for tv in P['stim']: ax.axvline(tv, color='gray', lw=0.6, ls=':')
                ax.set_title(f"C{ci+1}" + (" *" if ci == CENTRE_CLIQUE else ""), fontsize=9, loc='left')
                ax.spines[["top", "right"]].set_visible(False)
            axs[0, 0].legend(frameon=False, fontsize=8); fig.suptitle(f"{case}: S2 coupling interval {P['ci']:.0f} ms")
            fig.supxlabel("Time (ms)"); fig.supylabel("OEGM (all beats)")
            fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR, f"crn_fit_{case}_CI{P['ci']:.0f}_{GRAD_MODE}{FREE_TAG}{SMOOTH_TAG}.png"), dpi=150); plt.close(fig)
    except Exception as ex:
        print("  (figure skipped:", ex, ")")
    # ---- POST-HOC ONLY: MAPE against the truth json (never used by the optimiser) ------------
    tj = os.path.join(GT_DIR, f"egm_pipeline_for_jax_{case}", f"gt_crn_truth_{case}.json")
    mape = None
    if os.path.exists(tj):
        tr = json.load(open(tj)); tv = np.array([tr[n] for n in PARAM_NAMES])
        err = 100*np.abs(p_p2-tv)/np.abs(tv); mape = float(err[FREE_IDX].mean())
        print("\n  RESULT (truth used only here, after the fit):")
        print(f"    {'parameter':<10s}{'start':>9s}{'true':>9s}{'recovered':>11s}{'error %':>9s}")
        for i in FREE_IDX:
            print(f"    {LABELS[i]:<10s}{p_init[i]:>9.4f}{tv[i]:>9.4f}{p_p2[i]:>11.4f}{err[i]:>9.2f}")
        print(f"    mean abs. error over the {NFREE} recovered parameters: {mape:.2f}%")
        # ---- DIAGNOSTIC (post-hoc, truth json): is the minimum of the loss at the truth, and is the forward run repeatable? ----
        try:
            xt = np.array(inv_p6(jnp.array(tv, dtype=DTYPE)))[FREE_IDX]
            lt = [float(loss_p2_jit(jnp.array(xt, dtype=DTYPE))) for _ in range(3)]
            lf = [float(loss_p2_jit(jnp.array(xf, dtype=DTYPE))) for _ in range(3)]
            print(f"    DIAG full loss at the TRUTH (3 repeats): {['%.3e' % v for v in lt]}")
            print(f"    DIAG full loss at the FIT   (3 repeats): {['%.3e' % v for v in lf]}")
        except Exception as ex:
            print("    (diagnostic skipped:", ex, ")")
    print(f"  CASE {case} complete: {(time.time()-t_case)/60.:.1f} min")
    return dict(p=p_p2, loss=best_l2[0], evals=n_p2[0], mape=mape)

def main_inv():
    print(f"\n{'='*60}\nCRN PARAMETER ESTIMATION, 3 sequences (GT-free)   N_ION={N_ION}  DT_SIM={DT_SIM}  BETA={BETA}\n{'='*60}")
    print("CRN switches: " + ("published hard -40 mV switch" if SMOOTH_MV == 0. else f"-40 mV switch smoothed (sigmoid, {SMOOTH_MV:g} mV)")
          + f";  SR trigger width x{TRIG_W:g};  removable singularities: " + ("exact limits" if SAFE_POLES else "published guards")
          + f";  GRAD_MODE={GRAD_MODE};  GT file pattern={GT_FILE}")
    print(f"JAX devices: {jax.devices()}")
    res = {c: invert(c) for c in CASES}
    print(f"\n{'='*72}\nGRAND SUMMARY - CRN fit\n{'='*72}")
    for c, r in res.items():
        print(f"  {c:<6} loss={r['loss']:.2e} evals={r['evals']:3d} " + " ".join(f"{v:.3f}" for v in r['p'])
              + (f"   MAPE(post-hoc)={r['mape']:.2f}%" if r['mape'] is not None else ""))

main_inv()
