# -*- coding: utf-8 -*-
"""
crn.py
======
Courtemanche-Ramirez-Nattel (CRN, 1998) atrial ionic model in JAX, and a
JIT-compiled activation-time-map forward solver for the 3D thin-plate cube
that uses it (selected with IONIC_MODEL = "CRN" in the run scripts).

The ionic equations are the same as in
parameter_learning/3D_LA/crn_parameter_learning_patch_geometry.py
(only the conductances are arguments). Numerical options, as used in the paper:
  CRN_SMOOTH_MV = 0.5  sigmoid blend (width 0.5 mV) of the two published
                       branches of the I_Na h/j gates at -40 mV
                       (0 = published hard switch)
  CRN_SAFE_POLES = 1   removable singularities evaluated with their exact limits
  CRN_TRIG_W = 1       published width of the SR-release logistic triggers
Each can be overridden with an environment variable of the same name.

Coupling to the tissue (same as the CRN inversion script):
  * Strang splitting: N_ION ionic sub-steps (dt = DT/(2 N_ION)) -> Crank-Nicolson
    diffusion -> N_ION ionic sub-steps.
  * Diffusion acts on V_norm = (V + 81)/100 (fixed linear map).
  * Stimulus STIM_AMP (pA/pF) is applied inside the ionic sub-steps.

Parameter vector (6): p6 = [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL]
  m_* multiply the nominal CRN conductances; G_IL in mS/cm (G_IT = G_IL/4).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

DTYPE = jnp.float64

PNAMES_CRN = ['m_gNa', 'm_gto', 'm_gKr', 'm_gCaL', 'm_gK1', 'G_IL']
V_SCALE = 100.; V_SHIFT = -81.     # V_norm = (V - V_SHIFT)/V_SCALE

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

# ── Numerical options (defaults = settings used in the paper) ──────────────
SMOOTH_MV = float(os.environ.get("CRN_SMOOTH_MV", 0.5))
TRIG_W = float(os.environ.get("CRN_TRIG_W", 1.))
SAFE_POLES = os.environ.get("CRN_SAFE_POLES", "1") == "1"

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
# [2] ACTIVATION-TIME-MAP FORWARD SOLVER (CRN)
#     Same outputs and AT definitions as solver.make_forward_solver_atmap
#     (mMS), so every run/visualisation script works with either model.
# ══════════════════════════════════════════════════════════════════════════
def make_forward_solver_atmap_crn(Np, eu, ev, m_inv, mask_pace, ecot, ed_x,
                                  BETA, CM, DT, N_ION, STIM_AMP, sv_schedule,
                                  V_GATE=0.41, T_BLANK=None, DETECTION_WINDOW=None,
                                  EARLY_UPSTROKE_V=0.3, EARLY_UPSTROKE_SHARPNESS=20.0,
                                  CG_TOL=1e-6, CG_MAXITER=50):
    """
    Build a JIT-compiled CRN forward solver returning the activation-time map.

    run_forward_atmap(p6) -> (at_hard, at_soft, activated)
      p6 = [m_gNa, m_gto, m_gKr, m_gCaL, m_gK1, G_IL]

    V_GATE is in normalised units, V_norm = (V + 81)/100; the default 0.41
    corresponds to V = -40 mV. The soft AT uses the same dV-weighted,
    early-upstroke-restricted centroid as the mMS solver. The diffusion
    weights are built from p6[5] (G_IL) inside the traced function, so all
    six parameters are differentiable.
    """
    m_j = jnp.array(m_inv, dtype=DTYPE)
    eu_j = jnp.array(eu, dtype=jnp.int32)
    ev_j = jnp.array(ev, dtype=jnp.int32)
    mk_j = jnp.array(mask_pace, dtype=DTYPE)
    sv_j = jnp.array(sv_schedule)
    ecot_j = jnp.array(ecot, dtype=DTYPE)
    cos2_j = jnp.array(ed_x, dtype=DTYPE) ** 2

    if T_BLANK is None:
        first = int(np.argmax(sv_schedule)); j = first
        while j < len(sv_schedule) and sv_schedule[j]:
            j += 1
        T_BLANK = first * DT + (j - first) * DT + 1.0
    if DETECTION_WINDOW is None:
        DETECTION_WINDOW = 600.0

    NT = len(sv_schedule)
    t_arr = jnp.arange(NT, dtype=DTYPE) * DT
    dt_sub = DTYPE(DT / (2 * N_ION)); alpha = DTYPE(DT / 2.)
    to_norm = lambda Vm: (Vm - DTYPE(V_SHIFT)) / DTYPE(V_SCALE)
    to_mv = lambda Vn: Vn * DTYPE(V_SCALE) + DTYPE(V_SHIFT)

    @jax.jit
    def run_forward_atmap(p6):
        p6 = jnp.asarray(p6, dtype=DTYPE)
        gNa, gto, gKr, gCaL, gK1, gKs = conductances(p6)
        G_IL = p6[5]; G_IT = G_IL / DTYPE(4.)
        w = jnp.abs(ecot_j) * ((G_IL / (BETA * CM)) * cos2_j
                               + (G_IT / (BETA * CM)) * (DTYPE(1.) - cos2_j))

        def spmv(x):
            Kx = jnp.zeros(Np, dtype=x.dtype)
            Kx = Kx.at[eu_j].add(-w * x[ev_j] + w * x[eu_j])
            Kx = Kx.at[ev_j].add(-w * x[eu_j] + w * x[ev_j])
            return Kx

        def _cn(Vn):
            rhs = Vn - alpha * m_j * spmv(Vn)
            out, _ = jax.scipy.sparse.linalg.cg(
                lambda x: x + alpha * m_j * spmv(x), rhs, x0=Vn,
                tol=CG_TOL, maxiter=CG_MAXITER)
            return out

        def ion_half(state, I_s):
            return jax.lax.fori_loop(
                0, N_ION,
                lambda _, s: crn_ion_step(s, I_s, dt_sub, gNa, gto, gKr, gCaL, gK1, gKs),
                state)

        def scan_fn(carry, inputs):
            state, at_num, at_den, activated, at_hard = carry
            sv_t, t_now = inputs
            V_prev = to_norm(state[0])
            I_s = jnp.where(sv_t, DTYPE(STIM_AMP) * mk_j, jnp.zeros(Np, dtype=DTYPE))
            state = ion_half(state, I_s)
            state = (to_mv(_cn(to_norm(state[0]))),) + state[1:]
            state = ion_half(state, I_s)
            V = to_norm(state[0])
            in_beat1 = (t_now < DTYPE(DETECTION_WINDOW)).astype(DTYPE)
            early = jax.nn.sigmoid(-DTYPE(EARLY_UPSTROKE_SHARPNESS) * (V - DTYPE(EARLY_UPSTROKE_V)))
            w_at = jax.nn.relu(V - V_prev) * early * in_beat1
            at_num = at_num + t_now * w_at
            at_den = at_den + w_at
            fires = (~activated) & (V > DTYPE(V_GATE)) & \
                    (t_now > DTYPE(T_BLANK)) & (t_now < DTYPE(DETECTION_WINDOW))
            at_hard = jnp.where(fires, t_now, at_hard)
            activated = activated | fires
            return (state, at_num, at_den, activated, at_hard), None

        z = jnp.zeros(Np, dtype=DTYPE)
        carry0 = (crn_initial(Np), z, z, jnp.zeros(Np, dtype=jnp.bool_),
                  jnp.full(Np, DTYPE(DETECTION_WINDOW), dtype=DTYPE))
        (_, at_num, at_den, activated, at_hard), _ = jax.lax.scan(
            jax.checkpoint(scan_fn), carry0, (sv_j, t_arr))
        at_soft = at_num / (at_den + DTYPE(1e-6))
        return at_hard, at_soft, activated

    run_forward_atmap.g_il_differentiable = True
    return run_forward_atmap
