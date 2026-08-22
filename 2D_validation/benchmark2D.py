# -*- coding: utf-8 -*-

"""
JAX-EP vs openCARP benchmark on 20x7 mm flat plate.

JAX part:
dx=0.2 mm, dt=0.05 ms, BETA=140

LAT detection:
first V > V_GATE after T_BLANK
same logic as the JAX scan_fn carry

CARP part:
reads .igb Vm time series (download required in working dir. from : https://doi.org/10.5281/zenodo.22060549)
applies the same LAT criterion
reads mesh .pts for coordinates

Figure:
JAX LAT map
CARP LAT map
LAT difference map
node-wise scatter
LAT error histogram
"""

import os
import time
import re
from collections import defaultdict

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ============================================================================
# JAX CONFIGURATION
# ============================================================================

jax.config.update("jax_enable_x64", True)


# ============================================================================
# PATHS
# ============================================================================

# Repository directory containing this script.
REPO_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Input files are expected in the repository's benchmark data directory.
#
# Expected structure:
#
# 2D_validation/
# ├── benchmark2D.py
# └── benchmark/
#     ├── HD_Grid_20x7_200um.pts
#     ├── corner_stim.vtx
#     └── JAX_EP_benchmark.igb
#
CARP_MESH_DIR = os.path.join(
    REPO_DIR,
    "benchmark"
)

CARP_IGB_PATH = os.path.join(
    CARP_MESH_DIR,
    "JAX_EP_benchmark.igb"
)

CARP_MESH_NAME = "HD_Grid_20x7_200um"

# GitHub output folder.
OUT_DIR = os.path.join(
    REPO_DIR,
    "output"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================================
# PHYSICS
# ============================================================================

V_GATE = 0.13
A_CRIT = 0.13
BETA = 140.0
CM = 1.0

TAU_IN = 0.300
TAU_OUT = 5.000
TAU_OPEN = 120.0
TAU_CLOSE = 150.0

G_IL = 0.350
G_IT = G_IL / 4.0

STIM_AMP = 200.0
STIM_DUR = 2.0
N_ION = 2

LX = 20.0
LY = 7.0
TOTAL = 500.0
STIM_R = 1.5

DX = 0.2
DT = 0.05

T_BLANK = float(
    STIM_DUR + 2.0
)


# ============================================================================
# MESH
# ============================================================================

def build_mesh(dx_mm):
    """
    Read CARP .pts and triangulate by splitting each quadrilateral
    into two triangles.
    """

    pts_path = os.path.join(
        CARP_MESH_DIR,
        CARP_MESH_NAME + ".pts"
    )

    with open(pts_path) as f:

        n_carp = int(
            f.readline().strip()
        )

        carp_pts = np.array([
            list(
                map(
                    float,
                    f.readline().split()
                )
            )
            for _ in range(n_carp)
        ])

    # CARP coordinates are in micrometres.
    # Shift the minimum coordinate to zero and convert to mm.

    x_mm = (
        carp_pts[:, 0]
        - carp_pts[:, 0].min()
    ) * 1e-3

    y_mm = (
        carp_pts[:, 1]
        - carp_pts[:, 1].min()
    ) * 1e-3

    V = np.column_stack([
        x_mm,
        y_mm,
        np.zeros(n_carp)
    ])

    x_unique = np.unique(
        np.round(x_mm, 4)
    )

    y_unique = np.unique(
        np.round(y_mm, 4)
    )

    nx = len(x_unique)
    ny = len(y_unique)

    node_grid = np.arange(
        n_carp
    ).reshape(
        ny,
        nx
    )

    triangles = []

    for iy in range(ny - 1):

        for ix in range(nx - 1):

            bl = node_grid[iy, ix]
            br = node_grid[iy, ix + 1]
            tl = node_grid[iy + 1, ix]
            tr = node_grid[iy + 1, ix + 1]

            triangles.append([
                bl,
                br,
                tr
            ])

            triangles.append([
                bl,
                tr,
                tl
            ])

    E = np.array(
        triangles,
        dtype=np.int64
    )

    return V, E


# ============================================================================
# FEM
# ============================================================================

def build_fem(V_mm, E):

    Np = len(V_mm)

    # Convert mm to cm.
    vc = V_mm * 1e-1

    # Lumped nodal areas.
    ar = np.zeros(Np)

    for tri in E:

        i, j, k = tri

        cr = np.cross(
            vc[j] - vc[i],
            vc[k] - vc[i]
        )

        a = 0.5 * np.linalg.norm(cr)

        for nd in (
            i,
            j,
            k
        ):
            ar[nd] += a / 3.0

    # ------------------------------------------------------------------------
    # JAX FEM mass scaling
    # ------------------------------------------------------------------------

    m_j = jnp.array(
        1.0 / (
            CM
            * np.maximum(
                ar,
                max(
                    1e-12,
                    np.percentile(
                        ar[ar > 0],
                        5
                    )
                )
            )
        )
    )

    # ------------------------------------------------------------------------
    # Assemble edge conductivity contributions
    # ------------------------------------------------------------------------

    ec = defaultdict(float)

    for tri in E:

        i, j, k = tri

        pi = vc[i]
        pj = vc[j]
        pk = vc[k]

        for (
            u,
            v_,
            w,
            a_,
            b_
        ) in [
            (
                i,
                j,
                k,
                pj - pi,
                pk - pi
            ),
            (
                j,
                i,
                k,
                pi - pj,
                pk - pj
            ),
            (
                k,
                i,
                j,
                pi - pk,
                pj - pk
            )
        ]:

            c = np.linalg.norm(
                np.cross(
                    a_,
                    b_
                )
            )

            if c < 1e-14:
                continue

            ec[
                (
                    min(v_, w),
                    max(v_, w)
                )
            ] += (
                0.5
                * np.dot(a_, b_)
                / c
            )

    eu = np.array(
        [
            k[0]
            for k in ec
        ],
        dtype=np.int32
    )

    ev = np.array(
        [
            k[1]
            for k in ec
        ],
        dtype=np.int32
    )

    ecot = np.array(
        list(ec.values()),
        dtype=np.float64
    )

    # ------------------------------------------------------------------------
    # Anisotropic conductivity
    # ------------------------------------------------------------------------

    ed = vc[ev] - vc[eu]

    ed_n = ed / (
        np.linalg.norm(
            ed,
            axis=1,
            keepdims=True
        )
        + 1e-12
    )

    cos2 = ed_n[:, 0] ** 2

    w_aniso = (
        np.abs(ecot)
        * (
            (
                G_IL
                / (BETA * CM)
            )
            * cos2
            +
            (
                G_IT
                / (BETA * CM)
            )
            * (1.0 - cos2)
        )
    )

    return (
        m_j,
        jnp.array(
            eu,
            dtype=jnp.int32
        ),
        jnp.array(
            ev,
            dtype=jnp.int32
        ),
        jnp.array(
            w_aniso,
            dtype=jnp.float64
        )
    )


# ============================================================================
# SOLVER
# ============================================================================

def run_fem(V_mm, E, dt_ms):

    Np = len(V_mm)

    m_j, eu_j, ev_j, ew_j = build_fem(
        V_mm,
        E
    )

    # ------------------------------------------------------------------------
    # Stimulus
    # ------------------------------------------------------------------------

    VTX_PATH = os.path.join(
        CARP_MESH_DIR,
        "corner_stim.vtx"
    )

    with open(VTX_PATH) as f:

        n_vtx = int(
            f.readline()
        )

        # Skip second line.
        f.readline()

        vtx_nodes = [
            int(f.readline())
            for _ in range(n_vtx)
        ]

    mask = np.zeros(
        Np,
        dtype=np.float64
    )

    mask[vtx_nodes] = 1.0

    mk_j = jnp.array(
        mask,
        dtype=jnp.float64
    )

    ti_np = np.arange(
        int(TOTAL / dt_ms)
    )

    ti_ms = ti_np * dt_ms

    sv = (
        (ti_ms >= 10.0)
        &
        (
            ti_ms
            < 10.0 + STIM_DUR
        )
    )

    sv_j = jnp.array(sv)

    ti_j = jnp.arange(
        int(TOTAL / dt_ms),
        dtype=jnp.int32
    )

    dt_sub = (
        dt_ms
        / (2 * N_ION)
    )

    Iext = (
        (STIM_AMP / CM)
        * dt_ms
        * 1e-3
    )

    alpha = dt_ms / 2.0

    T_BLANK_LOCAL = 10.0

    # ------------------------------------------------------------------------
    # JAX solver
    # ------------------------------------------------------------------------

    @jax.jit
    def run():

        def spmv(x):

            Kx = jnp.zeros(
                Np,
                dtype=x.dtype
            )

            Kx = Kx.at[eu_j].add(
                ew_j
                * (
                    x[eu_j]
                    - x[ev_j]
                )
            )

            Kx = Kx.at[ev_j].add(
                ew_j
                * (
                    x[ev_j]
                    - x[eu_j]
                )
            )

            return Kx

        # --------------------------------------------------------------------
        # Mitchell-Schaeffer ionic update
        # --------------------------------------------------------------------

        def _ion(V, h):

            sw = jax.nn.sigmoid(
                jnp.float64(150.0)
                * (
                    V
                    - jnp.float64(
                        V_GATE
                    )
                )
            )

            dh = (
                (
                    (
                        jnp.float64(1.0)
                        - h
                    )
                    / jnp.float64(
                        TAU_OPEN
                    )
                )
                * (
                    jnp.float64(1.0)
                    - sw
                )
                -
                (
                    h
                    / jnp.float64(
                        TAU_CLOSE
                    )
                )
                * sw
            )

            Vn = jnp.clip(
                V
                + jnp.float64(dt_sub)
                * (
                    h
                    * V
                    * (
                        V
                        - jnp.float64(
                            A_CRIT
                        )
                    )
                    * (
                        jnp.float64(1.0)
                        - V
                    )
                    / jnp.float64(
                        TAU_IN
                    )
                    -
                    (
                        jnp.float64(1.0)
                        - h
                    )
                    * (
                        V
                        / jnp.float64(
                            TAU_OUT
                        )
                    )
                ),
                jnp.float64(0.0),
                jnp.float64(1.0)
            )

            hn = jnp.clip(
                h
                + jnp.float64(dt_sub)
                * dh,
                jnp.float64(0.0),
                jnp.float64(1.0)
            )

            return Vn, hn

        # --------------------------------------------------------------------
        # Crank-Nicolson diffusion step
        # --------------------------------------------------------------------

        def _cn(V):

            rhs = (
                V
                - jnp.float64(alpha)
                * m_j
                * spmv(V)
            )

            Vn, _ = (
                jax.scipy.sparse.linalg.cg(
                    lambda x:
                        x
                        + jnp.float64(alpha)
                        * m_j
                        * spmv(x),
                    rhs,
                    x0=V,
                    tol=1e-6,
                    maxiter=50
                )
            )

            return jnp.clip(
                Vn,
                jnp.float64(0.0),
                jnp.float64(1.0)
            )

        # --------------------------------------------------------------------
        # Time integration
        # --------------------------------------------------------------------

        def scan_fn(carry, inputs):

            V, h, act, at = carry

            sv_t, ti = inputs

            for _ in range(N_ION):

                V, h = _ion(
                    V,
                    h
                )

            V = jnp.where(
                sv_t,
                jnp.clip(
                    V
                    + jnp.float64(Iext)
                    * mk_j
                    * m_j,
                    jnp.float64(0.0),
                    jnp.float64(1.0)
                ),
                V
            )

            V = _cn(V)

            for _ in range(N_ION):

                V, h = _ion(
                    V,
                    h
                )

            t_now = (
                jnp.float64(ti)
                * jnp.float64(dt_ms)
            )

            fires = (
                (~act)
                &
                (
                    V
                    > jnp.float64(
                        V_GATE
                    )
                )
                &
                (
                    t_now
                    > jnp.float64(
                        T_BLANK_LOCAL
                    )
                )
            )

            at = jnp.where(
                fires,
                t_now,
                at
            )

            act = act | fires

            return (
                V,
                h,
                act,
                at
            ), None

        V0 = jnp.zeros(
            Np,
            dtype=jnp.float64
        )

        h0 = jnp.ones(
            Np,
            dtype=jnp.float64
        )

        a0 = jnp.zeros(
            Np,
            dtype=jnp.bool_
        )

        t0 = jnp.zeros(
            Np,
            dtype=jnp.float64
        )

        (
            _,
            _,
            actf,
            atf
        ), _ = jax.lax.scan(
            jax.checkpoint(
                scan_fn
            ),
            (
                V0,
                h0,
                a0,
                t0
            ),
            (
                sv_j,
                ti_j
            )
        )

        return (
            jnp.where(
                actf,
                atf,
                jnp.float64(TOTAL)
            ),
            actf
        )

    # ------------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------------

    t0 = time.time()

    atf, actf = run()

    at = np.array(
        atf,
        dtype=np.float64
    )

    act = np.array(
        actf,
        dtype=bool
    )

    elapsed = time.time() - t0

    # ------------------------------------------------------------------------
    # Conduction velocity
    # ------------------------------------------------------------------------

    y0 = V_mm[:, 1].min()

    xn = np.where(
        (
            np.abs(
                V_mm[:, 1] - y0
            )
            < 1.5
        )
        &
        (
            V_mm[:, 0] > 1.0
        )
    )[0]

    cv = np.nan

    if len(xn) > 2:

        ns = xn[
            np.argsort(
                V_mm[xn, 0]
            )
        ]

        ats = at[ns]

        valid = (
            (ats > 5.0)
            &
            (
                ats
                < TOTAL - 10.0
            )
        )

        if valid.sum() > 2:

            i1, i2 = (
                np.where(valid)[0][0],
                np.where(valid)[0][-1]
            )

            if (
                ats[i2] - ats[i1]
                > 0.5
            ):

                cv = float(
                    (
                        V_mm[ns[i2], 0]
                        - V_mm[ns[i1], 0]
                    )
                    * 0.1
                    * 10.0
                    / (
                        ats[i2]
                        - ats[i1]
                    )
                )

    return (
        at,
        act,
        cv
    )


# ============================================================================
# PART 1 — JAX RUN
# ============================================================================

V_jax, E_jax = build_mesh(
    DX
)

jax_at, jax_act, jax_cv = run_fem(
    V_jax,
    E_jax,
    DT
)

jax_at_plot = np.where(
    jax_act,
    jax_at,
    np.nan
)


# ============================================================================
# PART 2 — CARP
#
# Read .igb and apply the same LAT criterion.
#
# LAT:
# first timestep where
# V > V_GATE
# and
# t > T_BLANK
#
# CARP LAT is linearly interpolated between adjacent timesteps.
# ============================================================================

pts_path = os.path.join(
    CARP_MESH_DIR,
    CARP_MESH_NAME + ".pts"
)

with open(pts_path) as f:

    n_carp = int(
        f.readline().strip()
    )

    carp_pts = np.array([
        list(
            map(
                float,
                f.readline().split()
            )
        )
        for _ in range(n_carp)
    ])

# CARP coordinates:
# micrometres -> millimetres
# shifted so lower-left corner is (0,0)

carp_x = (
    carp_pts[:, 0]
    - carp_pts[:, 0].min()
) * 1e-3

carp_y = (
    carp_pts[:, 1]
    - carp_pts[:, 1].min()
) * 1e-3


# ---------------------------------------------------------------------------
# Read IGB
# ---------------------------------------------------------------------------

with open(
    CARP_IGB_PATH,
    "rb"
) as f:

    header = f.read(
        1024
    ).decode(
        "ascii",
        "ignore"
    )

    raw = f.read()


# ---------------------------------------------------------------------------
# Parse IGB dimensions
# ---------------------------------------------------------------------------

nx = re.findall(
    r"x:\s*(\d+)",
    header
)

nt = re.findall(
    r"t:\s*(\d+)",
    header
)

Np_igb = (
    int(nx[0])
    if nx
    else n_carp
)

NT_igb = (
    int(nt[0])
    if nt
    else int(
        TOTAL / DT
    )
)


# ---------------------------------------------------------------------------
# Reshape Vm data
# ---------------------------------------------------------------------------

vm_carp = np.frombuffer(
    raw,
    dtype=np.float32
)[
    :NT_igb * Np_igb
].reshape(
    NT_igb,
    Np_igb
).astype(
    np.float64
)


# ---------------------------------------------------------------------------
# CARP timestep
# ---------------------------------------------------------------------------

inc_t = re.findall(
    r"inc_t:\s*([0-9.eE+-]+)",
    header
)

dt_carp = (
    float(inc_t[0])
    if inc_t
    else DT
)


# ============================================================================
# CARP LAT DETECTION
# ============================================================================

T_BLANK_CARP = 10.0

t_carp = (
    np.arange(
        NT_igb
    )
    * dt_carp
)

carp_at = np.full(
    Np_igb,
    TOTAL,
    dtype=np.float64
)

carp_act = np.zeros(
    Np_igb,
    dtype=bool
)

for k in range(
    1,
    NT_igb
):

    if (
        t_carp[k]
        <= T_BLANK_CARP
    ):
        continue

    fires = (
        (~carp_act)
        &
        (
            vm_carp[
                k - 1,
                :
            ]
            < V_GATE
        )
        &
        (
            vm_carp[
                k,
                :
            ]
            >= V_GATE
        )
    )

    if fires.any():

        dv = (
            vm_carp[
                k,
                fires
            ]
            -
            vm_carp[
                k - 1,
                fires
            ]
        )

        frac = np.where(
            np.abs(dv) > 1e-12,
            (
                V_GATE
                -
                vm_carp[
                    k - 1,
                    fires
                ]
            )
            / dv,
            0.5
        )

        carp_at[fires] = (
            t_carp[k - 1]
            + frac * dt_carp
        )

        carp_act[fires] = True

    if carp_act.all():
        break

carp_at_plot = np.where(
    carp_act,
    carp_at,
    np.nan
)


# ============================================================================
# CARP CONDUCTION VELOCITY
# ============================================================================

cy0 = carp_y.min()

cxn = np.where(
    (
        np.abs(
            carp_y - cy0
        )
        < 0.5
    )
    &
    (
        carp_x > 1.0
    )
)[0]

carp_cv = np.nan

if len(cxn) > 2:

    cns = cxn[
        np.argsort(
            carp_x[cxn]
        )
    ]

    cats = carp_at[cns]

    cval = (
        (cats > 5.0)
        &
        (
            cats
            < TOTAL - 10.0
        )
    )

    if cval.sum() > 2:

        ci1, ci2 = (
            np.where(cval)[0][0],
            np.where(cval)[0][-1]
        )

        if (
            cats[ci2] - cats[ci1]
            > 0.5
        ):

            carp_cv = float(
                (
                    carp_x[cns[ci2]]
                    - carp_x[cns[ci1]]
                )
                * 0.1
                * 10.0
                / (
                    cats[ci2]
                    - cats[ci1]
                )
            )


# ============================================================================
# PART 3 — DIRECT NODE-TO-NODE COMPARISON
# ============================================================================

jax_at_valid = np.where(
    jax_act,
    jax_at,
    np.nan
)

carp_at_valid = np.where(
    carp_act,
    carp_at,
    np.nan
)

diff_direct = (
    jax_at_valid
    - carp_at_valid
)

valid_mask = (
    np.isfinite(
        jax_at_valid
    )
    &
    np.isfinite(
        carp_at_valid
    )
)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

l2_err = float(
    np.sqrt(
        np.nanmean(
            diff_direct[
                valid_mask
            ] ** 2
        )
    )
)

max_err = float(
    np.nanmax(
        np.abs(
            diff_direct[
                valid_mask
            ]
        )
    )
)

mean_err = float(
    np.nanmean(
        diff_direct[
            valid_mask
        ]
    )
)


# ============================================================================
# FIGURE DATA
# ============================================================================

triang = Triangulation(
    V_jax[:, 0],
    V_jax[:, 1],
    E_jax
)

jax_grid = jax_at_valid
carp_grid = carp_at_valid
diff_grid = diff_direct


# ============================================================================
# FIGURE
# ============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

at_min = min(
    np.nanmin(
        jax_grid[
            valid_mask
        ]
    ),
    np.nanmin(
        carp_grid[
            valid_mask
        ]
    )
)

at_max = max(
    np.nanmax(
        jax_grid[
            valid_mask
        ]
    ),
    np.nanmax(
        carp_grid[
            valid_mask
        ]
    )
)

iso_lev = np.arange(
    5,
    at_max,
    5
)

fig = plt.figure(
    figsize=(16, 12),
    facecolor="white"
)

ax_a = fig.add_axes(
    [0.04, 0.55, 0.27, 0.36]
)

ax_b = fig.add_axes(
    [0.37, 0.55, 0.27, 0.36]
)

ax_c = fig.add_axes(
    [0.70, 0.55, 0.27, 0.36]
)

ax_d = fig.add_axes(
    [0.04, 0.08, 0.42, 0.36]
)

ax_e = fig.add_axes(
    [0.57, 0.08, 0.38, 0.36]
)


def plbl(ax, label):

    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left"
    )


# ============================================================================
# A — JAX LAT
# ============================================================================

plbl(
    ax_a,
    "A"
)

tcf = ax_a.tripcolor(
    triang,
    jax_grid,
    cmap="jet",
    vmin=at_min,
    vmax=at_max,
    shading="gouraud"
)

try:

    cs = ax_a.tricontour(
        triang,
        jax_grid,
        levels=iso_lev,
        colors="white",
        linewidths=0.8,
        alpha=0.85
    )

    ax_a.clabel(
        cs,
        fmt="%d ms",
        fontsize=7,
        inline=True
    )

except Exception:
    pass

cb = plt.colorbar(
    tcf,
    ax=ax_a,
    fraction=0.03,
    pad=0.03
)

cb.set_label(
    "LAT (ms)",
    fontsize=8
)

cb.ax.tick_params(
    labelsize=7
)

ax_a.plot(
    0,
    0,
    "w*",
    ms=12,
    zorder=10
)

ax_a.set_xlabel(
    "x (mm)"
)

ax_a.set_ylabel(
    "y (mm)"
)

ax_a.set_title(
    f"JAX-EP LAT\nCV={jax_cv:.3f} m/s",
    fontweight="bold",
    pad=4
)

ax_a.set_aspect(
    "equal"
)

ax_a.set_xlim(
    0,
    LX
)

ax_a.set_ylim(
    0,
    LY
)


# ============================================================================
# B — CARP LAT
# ============================================================================

plbl(
    ax_b,
    "B"
)

tcf2 = ax_b.tripcolor(
    triang,
    carp_grid,
    cmap="jet",
    vmin=at_min,
    vmax=at_max,
    shading="gouraud"
)

try:

    cs2 = ax_b.tricontour(
        triang,
        carp_grid,
        levels=iso_lev,
        colors="white",
        linewidths=0.8,
        alpha=0.85
    )

    ax_b.clabel(
        cs2,
        fmt="%d ms",
        fontsize=7,
        inline=True
    )

except Exception:
    pass

cb2 = plt.colorbar(
    tcf2,
    ax=ax_b,
    fraction=0.03,
    pad=0.03
)

cb2.set_label(
    "LAT (ms)",
    fontsize=8
)

cb2.ax.tick_params(
    labelsize=7
)

ax_b.set_xlabel(
    "x (mm)"
)

ax_b.set_ylabel(
    "y (mm)"
)

ax_b.set_title(
    f"openCARP LAT\nCV={carp_cv:.3f} m/s",
    fontweight="bold",
    pad=4
)

ax_b.set_aspect(
    "equal"
)

ax_b.set_xlim(
    0,
    LX
)

ax_b.set_ylim(
    0,
    LY
)


# ============================================================================
# C — LAT DIFFERENCE
# ============================================================================

plbl(
    ax_c,
    "C"
)

dlim = max(
    abs(
        np.nanmin(
            diff_grid[
                valid_mask
            ]
        )
    ),
    abs(
        np.nanmax(
            diff_grid[
                valid_mask
            ]
        )
    )
) + 0.1

tcf3 = ax_c.tripcolor(
    triang,
    diff_grid,
    cmap="RdBu_r",
    vmin=-dlim,
    vmax=dlim,
    shading="gouraud"
)

cb3 = plt.colorbar(
    tcf3,
    ax=ax_c,
    fraction=0.03,
    pad=0.03
)

cb3.set_label(
    "JAX−CARP (ms)",
    fontsize=8
)

cb3.ax.tick_params(
    labelsize=7
)

ax_c.set_xlabel(
    "x (mm)"
)

ax_c.set_ylabel(
    "y (mm)"
)

ax_c.set_title(
    f"LAT difference (JAX−CARP)\n"
    f"L2={l2_err:.3f} ms  "
    f"Max={max_err:.3f} ms",
    fontweight="bold",
    pad=4
)

ax_c.set_aspect(
    "equal"
)

ax_c.set_xlim(
    0,
    LX
)

ax_c.set_ylim(
    0,
    LY
)


# ============================================================================
# D — SCATTER
# ============================================================================

plbl(
    ax_d,
    "D"
)

jf = jax_grid[
    valid_mask
]

cf = carp_grid[
    valid_mask
]

ax_d.scatter(
    cf,
    jf,
    s=2,
    alpha=0.4,
    color="#0072B2",
    rasterized=True
)

lims = [
    min(
        jf.min(),
        cf.min()
    ) - 1,

    max(
        jf.max(),
        cf.max()
    ) + 1
]

ax_d.plot(
    lims,
    lims,
    "k--",
    lw=1.0,
    label="Identity line"
)

ax_d.set_xlabel(
    "openCARP LAT (ms)"
)

ax_d.set_ylabel(
    "JAX-EP LAT (ms)"
)

ax_d.set_title(
    "JAX-EP vs openCARP LAT scatter\n"
    "(direct node-to-node)",
    fontweight="bold",
    pad=4
)

ax_d.legend(
    frameon=False,
    fontsize=8
)

ax_d.grid(
    lw=0.3,
    alpha=0.3
)


# ============================================================================
# E — HISTOGRAM
# ============================================================================

plbl(
    ax_e,
    "E"
)

err_flat = diff_grid[
    valid_mask
]

ax_e.hist(
    err_flat,
    bins=50,
    color="#0072B2",
    alpha=0.8,
    edgecolor="white"
)

ax_e.axvline(
    0,
    color="k",
    lw=1.0,
    ls="--",
    label="Zero"
)

ax_e.axvline(
    mean_err,
    color="#D55E00",
    lw=1.5,
    ls="--",
    label=(
        f"Mean bias = "
        f"{mean_err:.3f} ms"
    )
)

ax_e.set_xlabel(
    "JAX−CARP (ms)"
)

ax_e.set_ylabel(
    "Count"
)

ax_e.set_title(
    "LAT error distribution",
    fontweight="bold",
    pad=4
)

ax_e.legend(
    frameon=False,
    fontsize=8
)

ax_e.grid(
    lw=0.3,
    alpha=0.3
)


# ============================================================================
# SUMMARY BOX
# ============================================================================

summary = (
    f"L2 error:  {l2_err:.3f} ms\n"
    f"Max error: {max_err:.3f} ms\n"
    f"Mean bias: {mean_err:.3f} ms\n"
    f"JAX CV:    {jax_cv:.3f} m/s\n"
    f"CARP CV:   {carp_cv:.3f} m/s\n"
    f"CV diff:   "
    f"{abs(jax_cv - carp_cv) * 100:.2f}%\n"
    f"Nodes:     "
    f"{valid_mask.sum():,} (direct)"
)

ax_e.text(
    0.97,
    0.97,
    summary,
    transform=ax_e.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#F5F5F5",
        edgecolor="#CCCCCC",
        alpha=0.95
    )
)


# ============================================================================
# FIGURE TITLE
# ============================================================================

fig.text(
    0.5,
    0.975,
    "JAX-EP vs openCARP  |  20×7 mm flat plate  |  "
    "β=140 mm⁻¹  |  mMS ionic  |  "
    "dx=0.2mm  dt=0.05ms  |  "
    "LAT criterion: first V > 0.13 after blanking",
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
    color="#222222",
    transform=fig.transFigure
)


# ============================================================================
# SAVE FIGURE
# ============================================================================

fname = os.path.join(
    OUT_DIR,
    "figure_jax_vs_carp_benchmark.png"
)

fig.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)