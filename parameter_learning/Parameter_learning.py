# -*- coding: utf-8 -*-
"""
parameter_learning.py
==============================
GPU version of parameter_estimation.py (v5)
All logic identical to CPU version.
"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
from collections import defaultdict
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# ── Check GPU ────────────────────────────────────────────────────────────────
import subprocess
try:
    gpu_info=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total",
                             "--format=csv,noheader"],
                            capture_output=True,text=True).stdout.strip()
    print(f"GPU: {gpu_info}")
except: pass
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# ══════════════════════════════════════════════════════════════════════════
# [0] PATHS + SETTINGS  ← ONLY CHANGE FROM CPU VERSION
# ══════════════════════════════════════════════════════════════════════════
BASE_INPUT   = '/path_to_folder/data_upload'
CUBE_DIR     = BASE_INPUT
LABELLED_PTS = f'{BASE_INPUT}/Labelled.pts'
LABELLED_ELEM= f'{BASE_INPUT}/Labelled.elem'

CASES = ['set1', 'set2', 'set5']
PNAMES = ['tau_in','tau_out','tau_open','tau_close','G_IL']

# ── Physics ────────────────────────────────────────────────────────────────
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

# ── Bounds (logit space) ───────────────────────────────────────────────────
LB5=jnp.array([0.01,0.50, 80.0, 80.0,0.05],dtype=jnp.float64)
UB5=jnp.array([0.40,9.50,215.0,185.0,0.40],dtype=jnp.float64)

def get_p5(lp): return LB5+(UB5-LB5)*jax.nn.sigmoid(lp)
def inv_p5(p):
    ps=jnp.clip(jnp.array(p,dtype=jnp.float64),LB5+1e-6,UB5-1e-6)
    return jax.scipy.special.logit((ps-LB5)/(UB5-LB5))

# ── Phase 1 settings ──────────────────────────────────────────────────────
LOSS_TARGET_P1 = 0.0001
MAX_EVALS      = 250
PATIENCE       = 50
START_LEVEL    = 1

# ── Phase 2 settings ──────────────────────────────────────────────────────
FD_EPS_P2      = 1e-3
LOSS_STOP_P2   = 1e-6
GRAD_STOP_P2   = 1e-5
PLATEAU_WIN_P2 = 5
PLATEAU_TOL_P2 = 1e-3
MAX_EVALS_P2   = 100

# ── Initialisation ─────────────────────────────────────────────────────────
INIT_FRAC = 0.50   # midpoint of bounds (maximum uncertainty)

# ══════════════════════════════════════════════════════════════════════════
# [1] MESH
# ══════════════════════════════════════════════════════════════════════════
def read_pts(p):
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1)[:n,:3]
def read_elem(p):
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1,dtype=str)[:n,1:-1].astype(np.int64)

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
# [3] PATCH BUILDER
# ══════════════════════════════════════════════════════════════════════════
def build_patch(case):
    egm_dir=os.path.join(CUBE_DIR,f"egm_pipeline_for_jax_{case}")
    p1=np.load(os.path.join(egm_dir,f"part1_outputs_{case}.npz"),allow_pickle=True)
    p2=np.load(os.path.join(egm_dir,f"part2_outputs_{case}.npz"),allow_pickle=True)
    HD16=np.array(p1["HD16"]); fib3=np.array(p1["fib3"])
    cs_edge=str(p1["cs_edge"][0])
    pm=np.array(p2["params_nominal"],dtype=np.float64)
    DT_SIM=float(p2["DT_SIM"][0]); N_ION=int(p2["N_ION_SUB"][0])
    dth=float(pm[6]); G_IL_nom=float(np.exp(pm[4]))

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

    sig_r= 1 #G_IL_nom/(G_IL_nom+0.2)
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

    return run_3d, get_egm, DT_SIM, NT

# ══════════════════════════════════════════════════════════════════════════
# [4] MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
all_results = {}
print(f"\n{'='*60}")
print(f"PARAMETER ESTIMATION — GT-FREE (GPU)")
print(f"P1: NM S2-MSE | P2: FD L-BFGS-B S1+S2-MSE")
print(f"Initialisation: {INIT_FRAC*100:.0f}th percentile of bounds")
print(f"{'='*60}")

for CASE in CASES:
    t_case = time.time()
    print(f"\n{'='*60}")
    print(f"CASE: {CASE}")
    print(f"{'='*60}")

    egm_dir  = os.path.join(CUBE_DIR, f"egm_pipeline_for_jax_{CASE}")
    gt_data  = np.load(os.path.join(egm_dir, f"gt2_egm_{CASE}.npz"),
                       allow_pickle=True)

    segs_s2_gt = gt_data['segs_s2_gt']
    segs_s1_gt = gt_data['segs_s1_gt']
    p2ps_s2_gt = gt_data['p2ps_s2_gt']
    p2ps_s1_gt = gt_data['p2ps_s1_gt']
    DT_SIM_gt  = float(gt_data['DT_SIM'][0])
    i_s1       = int(gt_data['i_s1'][0])
    i_s2       = int(gt_data['i_s2'][0])
    win_s1     = int(gt_data['win_s1'][0])
    win_s2     = int(gt_data['win_s2'][0])
    ari_s1_gt  = float(gt_data['ari_s1_gt'][0])
    ari_s2_gt  = float(gt_data['ari_s2_gt'][0])
    at_s2_gt   = float(gt_data['at_s2_gt'][0])
    slew_s2_gt = float(gt_data['slew_s2_gt'][0])
    p2p_s2_gt  = float(gt_data['p2p_s2_gt'][0])

    print(f"  Loaded gt_egm_{CASE}.npz")
    print(f"  S2 window: {segs_s2_gt.shape}  S1 window: {segs_s1_gt.shape}")

    run_3d, get_egm, DT_SIM, NT = build_patch(CASE)
    assert abs(DT_SIM - DT_SIM_gt) < 1e-6

    run_3d_batch = jax.jit(jax.vmap(run_3d))

    def eval_one(x_np):
        p5_j=get_p5(jnp.array(x_np,dtype=jnp.float64))
        return get_egm(np.array(run_3d(p5_j)))

    def eval_batch(x_batch):
        p5b=jax.vmap(get_p5)(jnp.array(x_batch,dtype=jnp.float64))
        phi_b=np.array(run_3d_batch(p5b))
        return [get_egm(phi_b[k]) for k in range(phi_b.shape[0])]

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  PHASE 1 — Shape Fit (9 cliques, all 5 free)")
    print(f"  {'─'*56}")
    t_p1=time.time()

    def make_obj_from_egm(egms, level):
        egm_c5_=egms[CENTRE_CLIQUE]; losses={}
        l1=0.
        for ci in range(9):
            seg=egms[ci,i_s2:i_s2+win_s2]
            l1+=float(np.mean((seg-segs_s2_gt[ci])**2)/(p2ps_s2_gt[ci]**2))
        losses['l1_s2mse']=l1/9.
        if level>=2 and not np.isnan(ari_s1_gt):
            a=wyatt_ari(egm_c5_,i_s1,DT_SIM,win_s1)
            losses['l2_s1ari']=0. if np.isnan(a) else float(((a-ari_s1_gt)/ari_s1_gt)**2)
        if level>=3 and not np.isnan(ari_s2_gt):
            a=wyatt_ari(egm_c5_,i_s2,DT_SIM,win_s2)
            losses['l3_s2ari']=0. if np.isnan(a) else float(((a-ari_s2_gt)/ari_s2_gt)**2)
        if level>=4 and not np.isnan(at_s2_gt):
            a=get_at(egm_c5_,i_s2,DT_SIM)
            losses['l4_at']=0. if np.isnan(a) else float(((a-at_s2_gt)/(at_s2_gt+1e-9))**2)
        if level>=5 and not np.isnan(slew_s2_gt):
            s=get_slew(egm_c5_,i_s2,DT_SIM)
            losses['l5_slew']=0. if np.isnan(s) else float(((s-slew_s2_gt)/(slew_s2_gt+1e-9))**2)
        if level>=6:
            p=get_p2p(egm_c5_,i_s2,DT_SIM,win_s2)
            losses['l6_p2p']=0. if np.isnan(p) else float(((p-p2p_s2_gt)/(p2p_s2_gt+1e-9))**2)
        if level>=7:
            l7=0.
            for ci in range(9):
                seg=egms[ci,i_s1:i_s1+win_s1]
                l7+=float(np.mean((seg-segs_s1_gt[ci])**2)/(p2ps_s1_gt[ci]**2))
            losses['l7_s1mse']=l7/9.
        return float(np.sum(list(losses.values()))), losses

    def make_objective(level):
        def objective(x_np):
            return make_obj_from_egm(eval_one(x_np), level)
        return objective

    # Initialisation at INIT_FRAC percentile of bounds
    x_best = np.array(inv_p5(LB5 + INIT_FRAC*(UB5-LB5)), dtype=np.float64)
    converged = False

    p_init = np.array(get_p5(jnp.array(x_best, dtype=jnp.float64)))
    print(f"\n  Initialisation ({INIT_FRAC*100:.0f}th percentile of bounds):")
    for pn, pv in zip(PNAMES, p_init):
        print(f"    {pn:<12} = {float(pv):.4f}")

    for level in range(START_LEVEL, 8):
        if converged: break
        labels=['S2-MSE','S1-ARI','S2-ARI','AT','Slew','p2p','S1-MSE']
        print(f"\n  LEVEL {level}: "+' + '.join(labels[:level]))
        obj_fn=make_objective(level)

        step=1.5
        simplex=np.zeros((6,5),dtype=np.float64)
        simplex[0]=x_best.copy()
        for i in range(5): simplex[i+1]=x_best.copy(); simplex[i+1][i]+=step

        egms_simplex=eval_batch(simplex)
        losses_simplex=[make_obj_from_egm(e,level)[0] for e in egms_simplex]
        v0=losses_simplex[0]
        p_now=np.array(get_p5(jnp.array(x_best,dtype=jnp.float64)))
        print(f"  Start: loss={v0:.5f}  params={np.round(p_now,3)}")

        n_eval=[0]; best_x=[x_best.copy()]
        best_loss=[v0]; plateau_count=[0]
        t_start=time.time()

        def f_nm(x_np):
            v,_=obj_fn(x_np)
            p_=np.array(get_p5(jnp.array(x_np,dtype=jnp.float64)))
            n_eval[0]+=1
            if v<best_loss[0]-1e-7:
                best_loss[0]=v; best_x[0]=x_np.copy(); plateau_count[0]=0
            else: plateau_count[0]+=1
            if n_eval[0]%20==0 or n_eval[0]==1:
                print(f"  e={n_eval[0]:3d} loss={v:.5f} "
                      f"ti={p_[0]:.3f} to={p_[1]:.3f} "
                      f"top={p_[2]:.1f} tcl={p_[3]:.1f} G={p_[4]:.3f} "
                      f"t={time.time()-t_start:.0f}s",flush=True)
            if v<=LOSS_TARGET_P1:
                raise StopIteration(f"loss={v:.6f}<={LOSS_TARGET_P1}")
            if plateau_count[0]>=PATIENCE:
                raise StopIteration(f"Plateau {PATIENCE} evals")
            return v

        try:
            minimize(f_nm, x_best, method='Nelder-Mead',
                     options={'maxiter':500,'maxfev':MAX_EVALS,
                              'xatol':1e-4,'fatol':1e-6,
                              'initial_simplex':simplex,'disp':False})
            x_best=best_x[0]
        except StopIteration as e:
            msg=str(e)
            if 'Plateau' in msg: print(f"\n  → {msg}")
            else: print(f"\n  ✓ {msg}"); converged=True
            x_best=best_x[0]

        p_final=np.array(get_p5(jnp.array(x_best,dtype=jnp.float64)))
        print(f"  Level {level}: loss={best_loss[0]:.6f}  "
              f"params={np.round(p_final,3)}  evals={n_eval[0]}")
        if best_loss[0]<=LOSS_TARGET_P1:
            converged=True

    p_p1=np.array(get_p5(jnp.array(x_best,dtype=jnp.float64)))
    all_results[CASE]={'p_phase1':p_p1,'loss_p1':best_loss[0],
                       'x_logit':x_best.copy()}
    print(f"  Phase 1 done: loss={best_loss[0]:.6f}  "
          f"time={(time.time()-t_p1)/60.:.1f}min")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    print(f"  PHASE 2 — FD L-BFGS-B S1+S2 MSE (all 5 free)")
    print(f"  {'─'*58}")
    t_p2=time.time()

    x0_p2=all_results[CASE]['x_logit'].copy()
    p0_p2=np.array(get_p5(jnp.array(x0_p2,dtype=jnp.float64)))
    print(f"  Warm start: {np.round(p0_p2,3)}")

    def loss_s1s2_p2(p5_np):
        phi  = np.array(run_3d(jnp.array(p5_np,dtype=jnp.float64)))
        egms = get_egm(phi)
        l_s2 = float(np.mean([np.mean((egms[ci,i_s2:i_s2+win_s2]-segs_s2_gt[ci])**2)
                               /p2ps_s2_gt[ci]**2 for ci in range(9)]))
        l_s1 = float(np.mean([np.mean((egms[ci,i_s1:i_s1+win_s1]-segs_s1_gt[ci])**2)
                               /p2ps_s1_gt[ci]**2 for ci in range(9)]))
        return l_s2 + 0.5*l_s1

    def fd_grad_p2(lp5_np):
        g=np.zeros(5,dtype=np.float64)
        for i in range(5):
            lp_p=lp5_np.copy(); lp_p[i]+=FD_EPS_P2
            lp_m=lp5_np.copy(); lp_m[i]-=FD_EPS_P2
            p_p=np.array(get_p5(jnp.array(lp_p,dtype=jnp.float64)))
            p_m=np.array(get_p5(jnp.array(lp_m,dtype=jnp.float64)))
            g[i]=(loss_s1s2_p2(p_p)-loss_s1s2_p2(p_m))/(2*FD_EPS_P2)
        return g

    n_p2=[0]; best_loss_p2=[1e9]; best_x_p2=[x0_p2.copy()]
    loss_hist_p2=[]; t_p2s=time.time()

    def f_p2_fd(x_np):
        p5   =np.array(get_p5(jnp.array(x_np,dtype=jnp.float64)))
        v    =loss_s1s2_p2(p5)
        g_fd =fd_grad_p2(x_np)
        g_norm=float(np.linalg.norm(g_fd))
        n_p2[0]+=1
        if v<best_loss_p2[0]-1e-9: best_loss_p2[0]=v; best_x_p2[0]=x_np.copy()
        loss_hist_p2.append(v)
        print(f"  e={n_p2[0]:2d} loss={v:.7f} "
              f"ti={p5[0]:.3f} to={p5[1]:.3f} "
              f"top={p5[2]:.1f} tcl={p5[3]:.1f} G={p5[4]:.4f} "
              f"|g|={g_norm:.6f} t={time.time()-t_p2s:.0f}s",flush=True)
        if v<=LOSS_STOP_P2 and g_norm<=GRAD_STOP_P2:
            print(f"  ✓ Converged: loss={v:.2e} |g|={g_norm:.2e}")
            raise StopIteration
        if len(loss_hist_p2)>=PLATEAU_WIN_P2:
            recent=loss_hist_p2[-PLATEAU_WIN_P2:]
            rel_imp=(recent[0]-recent[-1])/(recent[0]+1e-12)
            if rel_imp<PLATEAU_TOL_P2 and g_norm<1e-4:
                print(f"  → Plateau: rel_imp={rel_imp:.2e}")
                raise StopIteration
        return v, g_fd

    try:
        minimize(f_p2_fd, x0_p2, method='L-BFGS-B', jac=True,
                 options={'maxiter':MAX_EVALS_P2,'maxfun':MAX_EVALS_P2*11,
                          'ftol':1e-12,'gtol':1e-6,
                          'maxcor':20,'disp':False})
    except StopIteration:
        pass

    xp2f  =best_x_p2[0].copy()
    p_p2  =np.array(get_p5(jnp.array(xp2f,dtype=jnp.float64)))

    all_results[CASE]['p_phase2'] =p_p2
    all_results[CASE]['loss_p2']  =best_loss_p2[0]
    all_results[CASE]['x_logit_p2']=xp2f.copy()
    all_results[CASE]['n_evals_p2']=n_p2[0]
    all_results[CASE]['time_min']  =(time.time()-t_case)/60.

    print(f"  Phase 2 done: loss={best_loss_p2[0]:.2e}  "
          f"evals={n_p2[0]}  time={(time.time()-t_p2)/60.:.1f}min")
    print(f"  Recovered: {np.round(p_p2,3)}")

    # Save results to /kaggle/working (writable)
    out_path=os.path.join('/kaggle/working', f"results_{CASE}.npz")
    np.savez(out_path,
        p_phase1  =all_results[CASE]['p_phase1'],
        p_phase2  =p_p2,
        loss_p1   =np.array([all_results[CASE]['loss_p1']]),
        loss_p2   =np.array([best_loss_p2[0]]),
        n_evals_p2=np.array([n_p2[0]]),
        time_min  =np.array([all_results[CASE]['time_min']]),
        x_logit_p2=xp2f,
        case      =np.array([CASE]),
    )
    print(f"  Saved: results_{CASE}.npz")

    t_total=(time.time()-t_case)/60.
    print(f"\n  {'═'*56}")
    print(f"  CASE {CASE} COMPLETE — {t_total:.1f}min")
    print(f"  Phase1 loss={all_results[CASE]['loss_p1']:.6f}  "
          f"Phase2 loss={best_loss_p2[0]:.2e}")
    print(f"  {'═'*56}")

# ── Grand summary ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("GRAND SUMMARY — GPU Run")
print(f"{'='*72}")
print(f"  {'Case':<8}{'P1 loss':>10}{'P2 loss':>10}{'Evals':>7}"
      f"{'Time':>8}  Recovered params")
for case,r in all_results.items():
    pf=r.get('p_phase2',r['p_phase1'])
    print(f"  {case:<8}{r['loss_p1']:>10.6f}"
          f"{r.get('loss_p2',float('nan')):>10.2e}"
          f"{r.get('n_evals_p2',0):>7}"
          f"{r.get('time_min',0):>7.1f}min  "
          +' '.join([f"{pf[i]:.3f}" for i in range(5)]))

# Results are saved to /path_to_dir/working/ — download from there
print("\nResults saved to /path_to_dir/working/:")
for case in CASES:
    fpath = os.path.join('/path_to_dir/working', f"results_{case}.npz")
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)/1e3
        print(f"  ✓ results_{case}.npz  ({size:.1f} KB)")
    else:
        print(f"  ✗ results_{case}.npz NOT FOUND")

print("\nDONE")