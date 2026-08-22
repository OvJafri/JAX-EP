# -*- coding: utf-8 -*-
"""
fem.py
======
FEM operator assembly for the 3D thin-plate cube: lumped nodal
areas, edge cotangent weights, and anisotropic conductivity weights.

Uses exactly the same cotangent-weight FE assembly as the 2D
flat-plate benchmark (the manuscript's own formally validated
convergence-study solver) and the patch/full-LA solvers elsewhere in
this codebase -- so results from the cube are directly comparable to
both.

Fibre direction convention matches the 2D benchmark: aligned along
the global x-axis (cos2 = component of each edge direction along x,
squared) -- appropriate for a flat/thin-plate geometry where a single
global fibre direction is well-defined, unlike the curved full-LA
surface which requires per-element fibre vectors (Fibre_l.lon).
"""
from collections import defaultdict
import numpy as np


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
