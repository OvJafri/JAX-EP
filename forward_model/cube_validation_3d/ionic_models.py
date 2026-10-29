# -*- coding: utf-8 -*-
"""
ionic_models.py
===============
One switch for the ionic model used by the forward scripts:

    IONIC_MODEL = "mMS"   modified Mitchell-Schaeffer (5 parameters)
    IONIC_MODEL = "CRN"   Courtemanche-Ramirez-Nattel 1998 (6 parameters)

MODEL_SETTINGS holds the per-model settings (stimulus, activation threshold,
nominal parameters). build_atmap_solver() returns the matching JIT-compiled
activation-time-map solver; both return (at_hard, at_soft, activated).
"""
import jax.numpy as jnp

from .fem import anisotropic_weights
from .solver import make_forward_solver_atmap
from .crn import make_forward_solver_atmap_crn, PNAMES_CRN

PNAMES_MMS = ['tau_in', 'tau_out', 'tau_open', 'tau_close', 'G_IL']

MODEL_SETTINGS = {
    "mMS": dict(
        PNAMES=PNAMES_MMS,
        P_NOMINAL=[0.300, 5.000, 120.0, 150.0, 0.350],
        STIM_AMP=200.0,     # normalised units (as in the original cube scripts)
        V_GATE=0.13,        # normalised V
        A_CRIT=0.13,
    ),
    "CRN": dict(
        PNAMES=PNAMES_CRN,
        P_NOMINAL=[1.0, 1.0, 1.0, 1.0, 1.0, 0.350],   # nominal CRN conductances, G_IL as mMS
        STIM_AMP=40.0,      # pA/pF, applied inside the ionic sub-steps. 20 does not
                            # capture from the one-node-wide edge at dx = 0.2 mm (tested)
        V_GATE=0.41,        # normalised V = (V+81)/100  ->  -40 mV
    ),
}


def get_model_settings(model):
    if model not in MODEL_SETTINGS:
        raise ValueError(f"IONIC_MODEL must be one of {list(MODEL_SETTINGS)}, got {model!r}")
    return MODEL_SETTINGS[model]


def build_atmap_solver(model, Np, eu, ev, m_inv, ecot, ed_x, mask_pace,
                       BETA, CM, DT, N_ION, sv_schedule, G_IL=None):
    """
    Return (run_forward_atmap, p_nominal, settings) for the chosen model.

    mMS: identical call to make_forward_solver_atmap as in the original
         scripts (diffusion weights built once at G_IL = P_NOMINAL[4]).
    CRN: make_forward_solver_atmap_crn (weights built from p6[5] inside).
    """
    s = get_model_settings(model)
    p_nom = list(s["P_NOMINAL"])
    if G_IL is not None:
        p_nom[-1] = G_IL
    if model == "mMS":
        w = anisotropic_weights(ecot, ed_x, G_IL=p_nom[-1], BETA=BETA, CM=CM)
        run = make_forward_solver_atmap(
            Np=Np, eu=eu, ev=ev, m_inv=m_inv, w=w, mask_pace=mask_pace,
            V_GATE=s["V_GATE"], A_CRIT=s["A_CRIT"], DT=DT, N_ION=N_ION,
            STIM_AMP=s["STIM_AMP"], CM=CM, sv_schedule=sv_schedule)
    else:
        run = make_forward_solver_atmap_crn(
            Np=Np, eu=eu, ev=ev, m_inv=m_inv, mask_pace=mask_pace,
            ecot=ecot, ed_x=ed_x, BETA=BETA, CM=CM, DT=DT, N_ION=N_ION,
            STIM_AMP=s["STIM_AMP"], sv_schedule=sv_schedule, V_GATE=s["V_GATE"])
    return run, jnp.array(p_nom, dtype=jnp.float64), s
