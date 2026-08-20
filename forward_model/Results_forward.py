# -*- coding: utf-8 -*-
"""
result_forward_simulation_v2.py
========================================================================
JAX-EP Left Atrial (LA) Forward Simulation & OEGM Visualization Pipeline

Generates publication-quality figure replicating manuscript findings:
  1. Panel A: Left lateral activation map under CS pacing with CV metrics.
  2. Panel B: Left lateral activation map under LAA pacing with CV metrics.
  3. Panel C: 9-clique Omnipolar Electrograms (OEGMs) from catheter grid.

Produces a consolidated 300 DPI multi-panel visualization layout (Fig. 2).
========================================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import butter, filtfilt

#BASE_DIR = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\validation\3D_cube"
BASE_DIR = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\Jax_EP\Results\forward_LA\forward_model_github_repo\data"

#OUT_DIR  = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\Jax_EP\Results"
OUT_DIR  = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\Jax_EP\Results\forward_LA\forward_model_github_repo\outputs"

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.linewidth': 0.8,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

CLIQUES = np.array([[1,2,5,6],[2,3,6,7],[3,4,7,8],
                    [5,6,9,10],[6,7,10,11],[7,8,11,12],
                    [9,10,13,14],[10,11,14,15],[11,12,15,16]])-1
CENTRE_CLIQUE = 4

def read_pts(p):
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1)[:n,:3]
def read_elem(p):
    with open(p) as f: f.readline()
    return np.loadtxt(p,skiprows=1,dtype=str)[:,1:-1].astype(np.int64)

print("Loading ...")
V = read_pts(os.path.join(BASE_DIR,"Labelled.pts"))
E = read_elem(os.path.join(BASE_DIR,"Labelled.elem"))
HD_all = read_pts(os.path.join(BASE_DIR,"HDgrid_cath.pts"))
HD_S1  = HD_all[:16,  :3]
HD_S2  = HD_all[16:32,:3]
print(f"  {len(V):,} nodes  {len(E):,} elements")

d_cs  = np.load(os.path.join(OUT_DIR,"la_forward_cs.npz"))
d_laa = np.load(os.path.join(OUT_DIR,"la_forward_laa.npz"))
at_cs   = np.array(d_cs['at_map'],  dtype=np.float32)
at_laa  = np.array(d_laa['at_map'], dtype=np.float32)
phi_cs  = np.array(d_cs['phi_T'],   dtype=np.float32)
phi_laa = np.array(d_laa['phi_T'],  dtype=np.float32)
t_ms    = np.array(d_cs['t_ms'],    dtype=np.float32)
cv_cs   = float(d_cs['cv'][0])
cv_laa  = float(d_laa['cv'][0])

CS_NODES  = np.array([12144,114179,111480,103207])
LAA_NODES = np.array([22167,128314,112734, 90795])
ctr_cs    = V[CS_NODES].mean(0)
ctr_laa   = V[LAA_NODES].mean(0)

MSTEP = max(1, len(E)//120000)
Es = E[::MSTEP]; Es = Es[Es.max(axis=1)<len(V)]
print(f"  {len(Es):,} triangles")

def compute_oegm(phi_np, HD16_, dt_ms=0.05):
    fs=1000./dt_ms; nyq=fs/2.
    b,a=butter(2,[30./nyq,min(450.,nyq*0.95)/nyq],btype='bandpass')
    phi_f=np.array([filtfilt(b,a,phi_np[:,e]) for e in range(16)])
    oegm=np.zeros((9,phi_f.shape[1]),dtype=np.float32)
    for ci,idx4 in enumerate(CLIQUES):
        p=HD16_[idx4,:]
        dx=max(np.linalg.norm(p[2]-p[0]),1e-9)
        dy=max(np.linalg.norm(p[1]-p[0]),1e-9)
        gx=((phi_f[idx4[2]]-phi_f[idx4[0]])+(phi_f[idx4[3]]-phi_f[idx4[1]]))/(2.*dx)
        gy=((phi_f[idx4[1]]-phi_f[idx4[0]])+(phi_f[idx4[3]]-phi_f[idx4[2]]))/(2.*dy)
        C=np.cov(np.stack([gx,gy]))+np.eye(2)*1e-9
        val,vec=np.linalg.eigh(C); dom=vec[:,np.argmax(val)]
        th=np.arctan2(dom[1],dom[0])
        o=gx*np.cos(th)+gy*np.sin(th)
        if abs(np.max(o))>abs(np.min(o)): o=-o
        oegm[ci]=o.astype(np.float32)
    return oegm

print("Computing EGMs ...")
oegm_cs  = compute_oegm(phi_cs,  HD_S1)
oegm_laa = compute_oegm(phi_laa, HD_S1)

CMAP     = 'jet'
C_CS     = '#0072B2'
C_LAA    = '#D55E00'
cs_cmin  = float(at_cs.min());  cs_cmax  = float(at_cs.max())
laa_cmin = float(at_laa.min()); laa_cmax = float(at_laa.max())

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.0, 20.0), facecolor='white')

# LA panels: top 50% — bottom raised to 0.49 to give space for legend
gs_la = gridspec.GridSpec(1, 2, figure=fig,
                          hspace=0.0, wspace=0.05,
                          top=0.97, bottom=0.49,
                          left=0.02, right=0.98)

# EGM panels: bottom 43% — top pulled down to 0.40 to give space for heading
gs_egm = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.42, wspace=0.28,
                           top=0.40, bottom=0.03,
                           left=0.07, right=0.97)

def add_la_panel(gs_idx, at_map, cmin, cmax,
                 pace_ctr, pace_col, title_top, show_cbar):
    ax = fig.add_subplot(gs_la[0, gs_idx], projection='3d')
    triang = Triangulation(V[:,0], V[:,1], Es)
    at_face = at_map[Es].mean(axis=1)
    surf = ax.plot_trisurf(triang, V[:,2],
                           cmap=CMAP, linewidth=0,
                           antialiased=True, alpha=0.95)
    surf.set_array(at_face)
    surf.set_clim(0, cmax)

    ax.scatter([pace_ctr[0]], [pace_ctr[1]], [pace_ctr[2]],
               c=pace_col, s=300, marker='*', zorder=10,
               edgecolors='white', linewidths=1.2)
    ax.scatter(HD_S1[:,0], HD_S1[:,1], HD_S1[:,2],
               c='white', s=55, marker='o', zorder=9,
               edgecolors='#222222', linewidths=0.8, alpha=1.0)
    ax.scatter(HD_S2[:,0], HD_S2[:,1], HD_S2[:,2],
               c='cyan', s=55, marker='o', zorder=9,
               edgecolors='#222222', linewidths=0.8, alpha=1.0)

    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    ax.dist = 5.5
    ax.view_init(elev=0, azim=-90)

    if show_cbar:
        norm = Normalize(vmin=0, vmax=cmax)
        sm   = ScalarMappable(norm=norm, cmap=CMAP)
        cb   = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.00,
                            orientation='vertical', format='%d')
        cb.set_label('Local Activation Time (ms)', fontsize=9,
                     rotation=270, labelpad=14)
        cb.ax.tick_params(labelsize=8)

    bbox = ax.get_position()
    fig.text((bbox.x0+bbox.x1)/2, bbox.y1+0.002,
             title_top, ha='center', va='bottom',
             fontsize=10, fontweight='bold',
             transform=fig.transFigure)
    return ax

# ── CS panel ──────────────────────────────────────────────────────────────
ax_cs = add_la_panel(
    0, at_cs, cs_cmin, cs_cmax,
    ctr_cs, 'black',
    f'CS Pacing  |  AT = [{cs_cmin:.0f}, {cs_cmax:.0f}] ms'
    f'  |  CV = {cv_cs:.3f} m/s  |  Left Lateral View',
    show_cbar=True)

bbox = ax_cs.get_position()
fig.text(bbox.x0+0.008, bbox.y1-0.008, 'A',
         fontsize=14, fontweight='bold', color='white',
         va='top', ha='left', transform=fig.transFigure,
         bbox=dict(facecolor='black', alpha=0.6, pad=2,
                   boxstyle='round,pad=0.2'))

# ── LAA panel ─────────────────────────────────────────────────────────────
ax_laa = add_la_panel(
    1, at_laa, laa_cmin, laa_cmax,
    ctr_laa, '#FF4500',
    f'LAA Pacing  |  AT = [{laa_cmin:.0f}, {laa_cmax:.0f}] ms'
    f'  |  CV = {cv_laa:.3f} m/s  |  Left Lateral View',
    show_cbar=True)

bbox = ax_laa.get_position()
fig.text(bbox.x0+0.008, bbox.y1-0.008, 'B',
         fontsize=14, fontweight='bold', color='white',
         va='top', ha='left', transform=fig.transFigure,
         bbox=dict(facecolor='black', alpha=0.6, pad=2,
                   boxstyle='round,pad=0.2'))

# ── Legend strip — sits just below the LA panels ──────────────────────────
fig.text(0.5, 0.55,
         u'\u2605 Pacing site     '
         u'\u25CF HD Grid Site 1 (white)     '
         u'\u25CF HD Grid Site 2 (cyan)     '
         u'Left Lateral View',
         ha='center', va='center', fontsize=8.5,
         transform=fig.transFigure, color='#333333')

# ── EGM section heading — sits between legend and EGM grid ───────────────
fig.text(0.5, 0.448,
         'C    Nine-Clique Omnipolar EGMs — HD Grid Site 1  '
         '(CS pacing: blue  |  LAA pacing: orange dashed)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         color='#111111', transform=fig.transFigure)

# ── 9-clique EGMs ─────────────────────────────────────────────────────────
i1  = min(len(t_ms), int(600./0.05))
t_w = t_ms[:i1]

for ci in range(9):
    r, c = ci//3, ci%3
    ax = fig.add_subplot(gs_egm[r, c])

    eg_cs  = oegm_cs[ci,  :i1]
    eg_laa = oegm_laa[ci, :i1]
    sc = max(np.ptp(eg_cs), np.ptp(eg_laa), 1e-9)

    ax.plot(t_w, eg_cs/sc,  color=C_CS,  lw=1.2, alpha=0.9)
    ax.plot(t_w, eg_laa/sc, color=C_LAA, lw=1.2, alpha=0.85, ls='--')
    ax.axvline(10., color='gray', lw=0.6, ls=':', alpha=0.6)

    is_c = (ci == CENTRE_CLIQUE)
    ax.set_facecolor('#EDF7ED' if is_c else 'white')
    lbl = f'C{ci+1}' + (u'  \u2605' if is_c else '')
    ax.set_title(lbl, fontsize=9, fontweight='bold',
                 color='darkgreen' if is_c else '#222222', pad=3)
    ax.set_xlim(t_w[0], t_w[-1])
    ax.tick_params(labelsize=7, length=2)
    ax.grid(lw=0.3, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if r == 2: ax.set_xlabel('Time (ms)', fontsize=8)
    else:      ax.set_xticklabels([])
    if c == 0: ax.set_ylabel('Normalised OEGM', fontsize=8)
    else:      ax.set_yticklabels([])
    if ci == 0:
        ax.legend(['CS pacing','LAA pacing'],
                  fontsize=7.5, frameon=False,
                  loc='upper right', ncol=1)

fname = os.path.join(OUT_DIR, 'result_forward_simulation_v2.png')
fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {fname}")
print("DONE")