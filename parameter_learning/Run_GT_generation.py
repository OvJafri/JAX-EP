# -*- coding: utf-8 -*-
"""
run_gt_generation.py
=====================
Ground truth Omnipolar EGM generation and saves the output gt_egm_{case}.npz

Outputs per case (gt_egm_{case}.npz):
  segs_s2_gt   (9, win_s2)  — S2 EGM windows, all 9 cliques
  segs_s1_gt   (9, win_s1)  — S1 EGM windows, all 9 cliques
  p2ps_s2_gt   (9,)         — S2 peak-to-peak normalisation
  p2ps_s1_gt   (9,)         — S1 peak-to-peak normalisation
  ari_s1_gt    scalar       — S1 ARI at centre clique (validation only)
  ari_s2_gt    scalar       — S2 ARI at centre clique (validation only)
  at_s2_gt     scalar       — S2 activation time (validation only)
  slew_s2_gt   scalar       — S2 max slew rate (validation only)
  p2p_s2_gt    scalar       — S2 peak-to-peak (validation only)
  DT_SIM       scalar       — timestep ms
  i_s1         int          — S1 start index
  i_s2         int          — S2 start index
  win_s1       int          — S1 window length
  win_s2       int          — S2 window length
  case         str          — case name

DATA PROVENANCE METHOD:
  - Base anatomy and connectivity are loaded from real world source files 
    (Labelled.pts, Labelled.elem, and Fibre_l.lon).
  - The entire forward electrophysiology time marching simulation is computed 
    completely in JAX (using jax.lax.scan, @jax.jit, and Crank-Nicolson solvers) 
    at the exact true parameter values.

NOTE: gt_params are NOT saved in the output file.
      The optimizer must never see them.
      MAPE is computed post-hoc only in visualisation.
"""

import os, time
import numpy as np
import jax, jax.numpy as jnp
from collections import defaultdict

jax.config.update("jax_enable_x64", True)

# ══════════════════════════════════════════════════════════════════════════
# [0] PATHS + GT PARAMS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR      = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\Jax_EP\Results\Parameter_recovery\Github_repo\input_data"
CUBE_DIR      = r"C:\Users\exx915\Documents\DERIstuff\Brompton_Projects\Jax_codes\Jax_EP\Results\Parameter_recovery\Github_repo\input_data"
LABELLED_PTS  = os.path.join(BASE_DIR, "Labelled.pts")
LABELLED_ELEM = os.path.join(BASE_DIR, "Labelled.elem")

# ── GT params — only used in THIS script, never passed to optimizer ────────
GT_SETS = {
    'set1': [0.340, 5.580, 192.0, 120.0, 0.200],
    'set2': [0.260, 3.460, 148.0, 176.0, 0.200],
    'set5': [0.300, 5.000, 120.0, 150.0, 0.200],
}
CASES = ['set1', 'set2', 'set5']

# ── Physics (must match optimizer exactly) ─────────────────────────────────
V_GATE=0.13; A_CRIT=0.13; BETA=100.; CM=1.
STIM_AMP=200.; STIM_DUR=2.
S1_START=10.; S2_BCL=500.
S2_ON=S1_START+S2_BCL; TOTAL_MS=S2_ON+600.

CLIQUES=np.array([[1,2,5,6],[2,3,6,7],[3,4,7,8],
                  [5,6,9,10],[6,7,10,11],[7,8,11,12],
                  [9,10,13,14],[10,11,14,15],[11,12,15,16]])-1
CENTRE_CLIQUE=4
EDGE_ELEC={"LEFT":np.array([0,4,8,12]),"RIGHT":np.array([3,7,11,15]),
           "BOTTOM":np.array([0,1,2,3]),"TOP":np.array([12,13,14,15])}
OFFSET_MM=3.0; STRIP_PAD_MM=2.0; BB_PAD=5000.

# ══════════════════════════════════════════════════════════════════════════
# [1] MESH
# ══════════════════════════════════════════════════════════════════════════
def read_pts(p):
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1)[:n,:3]
def read_elem(p):
    with open(p) as f: f.readline()
    return np.loadtxt(p,skiprows=1,dtype=str)[:,1:-1].astype(np.int64)

print("Loading mesh...")
Verts=read_pts(LABELLED_PTS); Elems=read_elem(LABELLED_ELEM)
print(f"  {len(Verts):,} nodes")

# ══════════════════════════════════════════════════════════════════════════
# [2] FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════
def wyatt_ari(egm, i_start, dt, win=4000):
    seg=egm[i_start:i_start+win]
    if len(seg)<10 or np.ptp(seg)<1e-9: return np.nan
    d1=np.gradient(seg,dt); grd=max(1,int(50./dt))
    i_at=int(np.argmin(d1))
    cap=min(len(d1),i_at+int(400./dt))
    if i_at+grd>=cap: return np.nan
    i_rt=i_at+grd+int(np.argmax(d1[i_at+grd:cap]))
    return float((i_rt-i_at)*dt)

def get_at(egm, i_start, dt, win=200):
    seg=egm[i_start:i_start+win]
    if len(seg)<5: return np.nan
    return float(np.argmin(np.gradient(seg,dt))*dt)

def get_slew(egm, i_start, dt, win=200):
    seg=egm[i_start:i_start+win]
    if len(seg)<5: return np.nan
    return float(np.max(np.abs(np.gradient(seg,dt))))

def get_p2p(egm, i_start, dt, win=4000):
    seg=egm[i_start:i_start+win]
    if len(seg)<5: return np.nan
    return float(np.ptp(seg))

# ══════════════════════════════════════════════════════════════════════════
# [3] PATCH BUILDER — identical to optimizer build_patch
# ══════════════════════════════════════════════════════════════════════════
def build_patch(case):
    egm_dir=os.path.join(CUBE_DIR,f"patch_geo")
    p1=np.load(os.path.join(egm_dir,f"patch_geo.npz"),allow_pickle=True)
    #p2=np.load(os.path.join(egm_dir,f"part2_outputs_{case}.npz"),allow_pickle=True)
    HD16=np.array(p1["HD16"]); fib3=np.array(p1["fib3"])
    cs_edge=str(p1["cs_edge"][0])
    #pm=np.array(p2["params_nominal"],dtype=np.float64)
    #DT_SIM=float(p2["DT_SIM"][0]); N_ION=int(p2["N_ION_SUB"][0])
    #dth=float(pm[6]); G_IL_nom=float(np.exp(pm[4]))
    
    
    DT_SIM = float(0.1)
    N_ION  = int(4)
    pm = np.array([np.log(GT_SETS[case][0]), np.log(GT_SETS[case][1]), np.log(GT_SETS[case][2]), np.log(GT_SETS[case][3]), np.log(GT_SETS[case][4]), np      .log(DT_SIM), 0.0], dtype=np.float64)
    dth = 0.0
    G_IL_nom = float(GT_SETS[case][4])


    bb0=np.vstack([HD16[CLIQUES[CENTRE_CLIQUE]],HD16[CLIQUES[3]]]).min(0)-BB_PAD
    bb1=np.vstack([HD16[CLIQUES[CENTRE_CLIQUE]],HD16[CLIQUES[3]]]).max(0)+BB_PAD
    msk=((Verts[:,0]>=bb0[0])&(Verts[:,0]<=bb1[0])&
         (Verts[:,1]>=bb0[1])&(Verts[:,1]<=bb1[1])&
         (Verts[:,2]>=bb0[2])&(Verts[:,2]<=bb1[2]))
    gids=np.where(msk)[0]; lv=Verts[gids]; Np=len(gids)
    rev={g:l for l,g in enumerate(gids)}
    le=np.array([[rev[n] for n in e] for e in Elems
                 if all(n in rev for n in e)],dtype=np.int64)
    vc=lv*1e-4
    print(f"  [{case}] {Np:,} nodes  pacing [{cs_edge}]:",end=' ')

    nn=np.zeros((Np,3)); ar=np.zeros(Np)
    for (i,j,k) in le:
        cr=np.cross(lv[j]-lv[i],lv[k]-lv[i]); a=0.5*np.linalg.norm(cr)
        n_=cr/(np.linalg.norm(cr)+1e-12)
        for nd in (i,j,k): ar[nd]+=a/3.; nn[nd]+=a*n_
    for i in range(Np):
        n=np.linalg.norm(nn[i])
        if n>1e-12: nn[i]/=n
    m_j=jnp.array(1./(CM*np.maximum(ar*1e-8,
            max(1e-8,np.percentile((ar*1e-8)[ar>0],5)))))

    ec=defaultdict(float)
    for (i,j,k) in le:
        pi,pj,pk=vc[i],vc[j],vc[k]
        for (u,v_,w,a_,b_) in [(i,j,k,pj-pi,pk-pi),(j,i,k,pi-pj,pk-pj),
                                (k,i,j,pi-pk,pj-pk)]:
            c=np.linalg.norm(np.cross(a_,b_))
            if c<1e-14: continue
            ec[(min(v_,w),max(v_,w))]+=0.5*np.dot(a_,b_)/c
    eu=np.array([u for u,v_ in ec.keys()],dtype=np.int32)
    ev=np.array([v_ for u,v_ in ec.keys()],dtype=np.int32)
    ecot=np.array(list(ec.values()),dtype=np.float64)
    ed=vc[ev]-vc[eu]; ed/=np.linalg.norm(ed,axis=1,keepdims=True)+1e-12
    en=(nn[eu]+nn[ev])/2.; en/=np.linalg.norm(en,axis=1,keepdims=True)+1e-12
    eu_j=jnp.array(eu); ev_j=jnp.array(ev)
    ecot_j=jnp.array(ecot); ed_j=jnp.array(ed); en_j=jnp.array(en)
    fib3_j=jnp.array(np.tile(fib3,(len(eu),1)) if fib3.ndim==1 else fib3)

    ep=HD16[EDGE_ELEC[cs_edge]]; ec2=ep.mean(0)
    c5c=HD16[CLIQUES[CENTRE_CLIQUE]].mean(0)
    wf=(c5c-ec2)/(np.linalg.norm(c5c-ec2)+1e-12); lc=ec2-wf*OFFSET_MM*1000.
    _,_,Ve=np.linalg.svd(ep-ec2,full_matrices=False); ld=Ve[0]
    projs=(ep-ec2)@ld; hl=max(np.ptp(projs)/2.,1.)+2000.
    d=lv-lc; pal=d@ld; perp=d-np.outer(pal,ld)
    dist=np.linalg.norm(perp,axis=1)
    mk=(dist<=STRIP_PAD_MM*1000.)&(np.abs(pal)<=hl)
    if mk.sum()<10: mk=(dist<=STRIP_PAD_MM*2000.)&(np.abs(pal)<=hl+2000.)
    mk_j=jnp.array(mk.astype(np.float64))
    print(f"{int(mk.sum())} nodes")

    sig_r=1 #G_IL_nom/(G_IL_nom+0.2)
    W_np=np.zeros((16,Np))
    for e_ in range(16):
        r_=np.maximum(np.linalg.norm(vc-HD16[e_]*1e-4,axis=1),1e-6)
        W_np[e_]=(1./r_)*(sig_r/(4.*np.pi))
    W_j=jnp.array(W_np)

    NT=int(TOTAL_MS/DT_SIM); t_ms=np.arange(NT)*DT_SIM
    sv=np.zeros(NT,dtype=bool)
    sv|=(t_ms>=S1_START)&(t_ms<S1_START+STIM_DUR)
    sv|=(t_ms>=S2_ON)&(t_ms<S2_ON+STIM_DUR)
    sv_j=jnp.array(sv)

    i_s1=int(S1_START/DT_SIM)
    i_s2=int(S2_ON/DT_SIM)
    win_s1=int(S2_BCL/DT_SIM)
    win_s2=min(int(400./DT_SIM), NT-i_s2)

    def _build_w(g_il):
        g_it=g_il/4.
        fs=fib3_j-jnp.sum(fib3_j*en_j,axis=1,keepdims=True)*en_j
        fs/=jnp.linalg.norm(fs,axis=1,keepdims=True)+1e-12
        pp=jnp.cross(en_j,fs); pp/=jnp.linalg.norm(pp,axis=1,keepdims=True)+1e-12
        fr=fs*jnp.cos(dth)+pp*jnp.sin(dth)
        fr/=jnp.linalg.norm(fr,axis=1,keepdims=True)+1e-12
        cos2=jnp.sum(ed_j*fr,axis=1)**2
        return jnp.abs(ecot_j)*((g_il/(BETA*CM))*cos2+(g_it/(BETA*CM))*(1.-cos2))

    @jax.jit
    def run_3d(p5_j):
        g_il=p5_j[4]; w=_build_w(g_il)
        dt_sub=DT_SIM/(2*N_ION); Iext=(STIM_AMP/CM)*DT_SIM*1e-3; alpha=DT_SIM/2.
        def spmv(x):
            Kx=jnp.zeros(Np,dtype=x.dtype)
            Kx=Kx.at[eu_j].add(-w*x[ev_j]+w*x[eu_j])
            Kx=Kx.at[ev_j].add(-w*x[eu_j]+w*x[ev_j])
            return Kx
        def _ion(V,h):
            sw=jax.nn.sigmoid(150.*(V-V_GATE))
            dh=((1.-h)/p5_j[2])*(1.-sw)-(h/p5_j[3])*sw
            return (jnp.clip(V+dt_sub*((h*V*(V-A_CRIT)*(1.-V))/p5_j[0]
                                       -(1.-h)*(V/p5_j[1])),0.,1.),
                    jnp.clip(h+dt_sub*dh,0.,1.))
        def _cn(V):
            rhs=V-alpha*m_j*spmv(V)
            Vn,_=jax.scipy.sparse.linalg.cg(
                lambda x:x+alpha*m_j*spmv(x),rhs,x0=V,tol=1e-6,maxiter=50)
            return jnp.clip(Vn,0.,1.)
        def scan_fn(c,sv_):
            V,h=c
            for _ in range(N_ION): V,h=_ion(V,h)
            V=jnp.where(sv_,jnp.clip(V+Iext*mk_j,0.,1.),V); V=_cn(V)
            for _ in range(N_ION): V,h=_ion(V,h)
            return (V,h),W_j@V
        V0=jnp.zeros(Np,dtype=jnp.float64); h0=jnp.ones(Np,dtype=jnp.float64)
        (_,_),phi_T=jax.lax.scan(jax.checkpoint(scan_fn),(V0,h0),sv_j)
        return phi_T

    def get_egm(phi):
        egms=[]
        for ci in range(9):
            c_=CLIQUES[ci]
            egms.append((phi[:,c_[0]]-phi[:,c_[1]]
                         -phi[:,c_[2]]+phi[:,c_[3]])/4.)
        return np.array(egms)

    return run_3d, get_egm, DT_SIM, i_s1, i_s2, win_s1, win_s2, NT

# ══════════════════════════════════════════════════════════════════════════
# [4] PART 3b — GT EGM GENERATION
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART 3b — GT EGM Generation")
print(f"{'='*60}")

for case in CASES:
    print(f"\n{'─'*60}")
    print(f"Case: {case}")
    print(f"{'─'*60}")

    gt = GT_SETS[case]   # only used here

    # Build patch
    run_3d, get_egm, DT_SIM, i_s1, i_s2, win_s1, win_s2, NT = build_patch(case)

    # Run forward at GT params
    print(f"  Running 3D forward at GT params...")
    t0=time.time()
    phi_gt  = np.array(run_3d(jnp.array(gt, dtype=jnp.float64)))
    egms_gt = get_egm(phi_gt)
    egm_c5  = egms_gt[CENTRE_CLIQUE]
    print(f"  Done {time.time()-t0:.1f}s")

    # Extract targets
    segs_s2_gt = egms_gt[:, i_s2:i_s2+win_s2]
    segs_s1_gt = egms_gt[:, i_s1:i_s1+win_s1]
    p2ps_s2_gt = np.array([float(np.ptp(segs_s2_gt[ci]))+1e-9 for ci in range(9)])
    p2ps_s1_gt = np.array([float(np.ptp(segs_s1_gt[ci]))+1e-9 for ci in range(9)])

    # Validation-only features (NOT used in optimizer)
    ari_s1_gt  = wyatt_ari(egm_c5, i_s1, DT_SIM, win_s1)
    ari_s2_gt  = wyatt_ari(egm_c5, i_s2, DT_SIM, win_s2)
    at_s2_gt   = get_at(egm_c5, i_s2, DT_SIM)
    slew_s2_gt = get_slew(egm_c5, i_s2, DT_SIM)
    p2p_s2_gt  = get_p2p(egm_c5, i_s2, DT_SIM, win_s2)

    print(f"  S1 ARI={ari_s1_gt:.1f}ms  S2 ARI={ari_s2_gt:.1f}ms")
    print(f"  AT={at_s2_gt:.1f}ms  Slew={slew_s2_gt:.2f}  p2p={p2p_s2_gt:.2f}")

    # Save — no GT params in file
    out_dir = os.path.join(CUBE_DIR, f"egm_pipeline_for_jax_{case}")
    out_path = os.path.join(out_dir, f"gt2_egm_{case}.npz")
    np.savez(out_path,
        segs_s2_gt  = segs_s2_gt.astype(np.float64),
        segs_s1_gt  = segs_s1_gt.astype(np.float64),
        p2ps_s2_gt  = p2ps_s2_gt.astype(np.float64),
        p2ps_s1_gt  = p2ps_s1_gt.astype(np.float64),
        ari_s1_gt   = np.array([ari_s1_gt]),   # validation only
        ari_s2_gt   = np.array([ari_s2_gt]),   # validation only
        at_s2_gt    = np.array([at_s2_gt]),    # validation only
        slew_s2_gt  = np.array([slew_s2_gt]), # validation only
        p2p_s2_gt   = np.array([p2p_s2_gt]),  # validation only
        DT_SIM      = np.array([DT_SIM]),
        i_s1        = np.array([i_s1]),
        i_s2        = np.array([i_s2]),
        win_s1      = np.array([win_s1]),
        win_s2      = np.array([win_s2]),
        case        = np.array([case]),
    )
    print(f"  Saved: {out_path}")
    print(f"  segs_s2_gt: {segs_s2_gt.shape}  segs_s1_gt: {segs_s1_gt.shape}")

print(f"\n{'='*60}")
print("PART 3b COMPLETE — GT EGMs saved for all cases")
print("GT params NOT saved — optimizer has no access to them")
print(f"{'='*60}")