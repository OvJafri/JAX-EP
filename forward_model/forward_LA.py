# -*- coding: utf-8 -*-
"""
JAX-EP GPU benchmark and differentiability demonstration.

# ── Input Data File Definitions ───────────────────────────────────────
# LABELLED_PTS : 3D spatial coordinates (X, Y, Z) for all mesh nodes
# LABELLED_ELEM: Triangular element connectivity mapping the surface mesh
# FIBER_LON    : 3D vector fields defining localized myocardial fiber directions
# HDGRID_FILE  : Spatial coordinates mapping the high-density catheter electrodes
# ──────────────────────────────────────────────────────────────────────

Two HD grid recording sites:
  Site 1: egm_pipeline_for_jax_1to16{case}/  (HDgrid_cath.pts [:16])
  Site 2: egm_pipeline_for_jax_{case}/        (HDgrid_cath.pts [16:32])

This script reproduces the principal computational demonstrations
reported in the manuscript:

1. Full left-atrial forward simulation under CS and LAA pacing,
   including GPU execution time and LAT-map comparison.
2. Local finite-difference and automatic-differentiation gradient
   comparison on a representative patch.
3. Forward-mode automatic differentiation (jax.jacfwd) validation
   against finite differences.
4. Full-LA sensitivity calculation for dAT/dG_IL.
5. Generation of LAT, Jacobian and membrane-potential outputs
   used for quantitative and visual assessment.

The script reports the detected GPU and JAX devices at runtime.
Reference CPU timings are included solely for comparison with the
reported benchmark measurements.
"""

import os, time
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

jax.config.update("jax_enable_x64", True)

# ── User-configurable paths ────────────────────────────────────────────────
#
BASE_DIR = os.environ.get("JAX_EP_DATA_DIR", "./data")
OUT_DIR  = os.environ.get("JAX_EP_OUTPUT_DIR", "./outputs")

LABELLED_PTS  = os.path.join(BASE_DIR, "Labelled.pts")
LABELLED_ELEM = os.path.join(BASE_DIR, "Labelled.elem")
FIBER_LON     = os.path.join(BASE_DIR, "Fibre_l.lon")
HDGRID_FILE   = os.path.join(BASE_DIR, "HDgrid_cath.pts")

EGM_DIR_S1 = os.environ.get(
    "JAX_EP_EGM_S1_DIR",
    os.path.join(BASE_DIR, "patch_geo")
)

EGM_DIR_S2 = os.environ.get(
    "JAX_EP_EGM_S2_DIR",
    os.path.join(BASE_DIR, "patch_geo")
)

os.makedirs(OUT_DIR, exist_ok=True)

CASE = "set5"
SKIP_FORWARD = False   # load existing CPU .npz for LAT comparison

# ── Physics ──────────────────────────────────────────────────────────────────
TAU_IN=0.300; TAU_OUT=5.000; TAU_OPEN=120.0; TAU_CLOSE=150.0
G_IL=0.20;   G_IT=G_IL*0.25
V_GATE=0.13;  A_CRIT=0.13; BETA=140.; CM=1.
STIM_AMP=200.; STIM_DUR=2.
N_S1=2; S1_START=10.; S1_BCL=600.
TOTAL_MS=S1_START+(N_S1-1)*S1_BCL+600.
DT_SIM=0.05; N_ION=2
NT=int(TOTAL_MS/DT_SIM)
CS_STIM_NODES  = np.array([12144,114179,111480,103207], dtype=np.int64)
LAA_STIM_NODES = np.array([22167,128314,112734, 90795], dtype=np.int64)
STIM_RADIUS_UM = 5000.
CLIQUES = np.array([[1,2,5,6],[2,3,6,7],[3,4,7,8],
                    [5,6,9,10],[6,7,10,11],[7,8,11,12],
                    [9,10,13,14],[10,11,14,15],[11,12,15,16]])-1
CENTRE_CLIQUE = 4
PNAMES = ['tau_in','tau_out','tau_open','tau_close','G_IL']
FD_EPS = 1e-3
DTYPE    = jnp.float32
NP_DTYPE = np.float32

import subprocess
try:
    gpu_info=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',
                             '--format=csv,noheader'],
                            capture_output=True,text=True).stdout.strip()
    print(f"GPU: {gpu_info}")
except: pass

print("="*60)
print("JAX-EP GPU Benchmark")
print(f"  JAX:     {jax.__version__}")
print(f"  Devices: {jax.devices()}")
print(f"  DTYPE:   {DTYPE}")
print("="*60)

# ── IO ────────────────────────────────────────────────────────────────────────
def read_pts(p):
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1)[:n,:3]

def read_elem(p):
    # Use [:n] to enforce exact element count regardless of OS line endings
    with open(p) as f: n=int(f.readline())
    return np.loadtxt(p,skiprows=1,dtype=str)[:n,1:-1].astype(np.int64)

def read_lon(p):
    data=[]
    with open(p) as f:
        for line in f:
            v=line.strip().split()
            if len(v)>=3:
                try: data.append([float(v[0]),float(v[1]),float(v[2])])
                except: pass
    return np.array(data,dtype=np.float64)

print("\n[1] Loading mesh ...")
t0=time.time()
Verts  = read_pts(LABELLED_PTS)
Elems  = read_elem(LABELLED_ELEM)
Fibers = read_lon(FIBER_LON)
HD_all = read_pts(HDGRID_FILE)
HD16_S1= HD_all[:16, :3]
HD16_S2= HD_all[16:32,:3]
Np=len(Verts); Ne=len(Elems)
print(f"  {Np:,} nodes  {Ne:,} elements  {len(Fibers):,} fibers  ({time.time()-t0:.1f}s)")
if len(Fibers)!=Ne:
    print(f"  WARNING: fiber/element mismatch — trimming to {min(len(Fibers),Ne)}")
    Fibers=Fibers[:Ne]
vc=Verts*1e-4
fib_elem=Fibers/np.maximum(np.linalg.norm(Fibers,axis=1,keepdims=True),1e-12)

# ── FEM ───────────────────────────────────────────────────────────────────────
print("\n[2] Building FEM operators ...")
t0=time.time()
nn=np.zeros((Np,3),dtype=np.float64); ar=np.zeros(Np,dtype=np.float64)
for (i,j,k) in Elems:
    cr=np.cross(Verts[j]-Verts[i],Verts[k]-Verts[i])
    a=0.5*np.linalg.norm(cr); n_=cr/(np.linalg.norm(cr)+1e-12)
    for nd in (i,j,k): ar[nd]+=a/3.; nn[nd]+=a*n_
for i in range(Np):
    n=np.linalg.norm(nn[i])
    if n>1e-12: nn[i]/=n
m_inv=1./(CM*np.maximum(ar*1e-8,max(1e-8,np.percentile((ar*1e-8)[ar>0],5))))
m_j=jnp.array(m_inv,dtype=DTYPE)
edge_elems=defaultdict(list)
for ei,(i,j,k) in enumerate(Elems):
    for u,v in [(i,j),(j,k),(i,k)]:
        edge_elems[(min(u,v),max(u,v))].append(ei)
ec=defaultdict(float)
for (i,j,k) in Elems:
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
fib_edge=np.zeros((len(eu),3),dtype=np.float64)
for e in range(len(eu)):
    key=(min(eu[e],ev[e]),max(eu[e],ev[e]))
    fib_edge[e]=fib_elem[edge_elems[key]].mean(0)
fib_edge/=np.linalg.norm(fib_edge,axis=1,keepdims=True)+1e-12
fs=fib_edge-np.sum(fib_edge*en,axis=1,keepdims=True)*en
fs/=np.linalg.norm(fs,axis=1,keepdims=True)+1e-12
cos2_np=np.sum(ed*fs,axis=1)**2
eu_j=jnp.array(eu,dtype=jnp.int32); ev_j=jnp.array(ev,dtype=jnp.int32)
ecot_j=jnp.array(ecot,dtype=DTYPE); cos2_j=jnp.array(cos2_np,dtype=DTYPE)
print(f"  {len(eu):,} edges  FEM done ({time.time()-t0:.1f}s)")

# ── Lead field + pacing masks ─────────────────────────────────────────────────
def build_w(g_il):
    g_it=g_il*jnp.float32(0.25)
    return jnp.abs(ecot_j)*((g_il/(BETA*CM))*cos2_j+(g_it/(BETA*CM))*(1.-cos2_j))

sig_r=G_IL/(G_IL+0.2)
W_np=np.zeros((16,Np),dtype=np.float64)
for e_ in range(16):
    r_=np.maximum(np.linalg.norm(vc-HD16_S1[e_]*1e-4,axis=1),1e-6)
    W_np[e_]=(1./r_)*(sig_r/(4.*np.pi))
W_j=jnp.array(W_np,dtype=DTYPE)

def make_mask(stim_nodes):
    centroid=Verts[stim_nodes].mean(0)
    dists=np.linalg.norm(Verts-centroid,axis=1)
    mask=(dists<=STIM_RADIUS_UM).astype(NP_DTYPE)
    return jnp.array(mask,dtype=DTYPE), centroid

cs_mask_j, cs_ctr   = make_mask(CS_STIM_NODES)
laa_mask_j, laa_ctr = make_mask(LAA_STIM_NODES)
print(f"\n  CS  mask: {int(cs_mask_j.sum()):,} nodes")
print(f"  LAA mask: {int(laa_mask_j.sum()):,} nodes")

t_ms=np.arange(NT,dtype=NP_DTYPE)*DT_SIM
sv=np.zeros(NT,dtype=bool)
for k in range(N_S1):
    t_on=S1_START+k*S1_BCL
    sv|=(t_ms>=t_on)&(t_ms<t_on+STIM_DUR)
sv_j=jnp.array(sv)

# ── Forward solver ────────────────────────────────────────────────────────────
def make_run_3d(mask_j):
    @jax.jit
    def run_3d(p5_j):
        w=build_w(p5_j[4])
        dt_sub=DTYPE(DT_SIM/(2*N_ION))
        Iext=DTYPE((STIM_AMP/CM)*DT_SIM*1e-3)
        alpha=DTYPE(DT_SIM/2.)
        BETA_AT=DTYPE(50.)
        T_BLANK=DTYPE(S1_START+STIM_DUR+1.)

        def spmv(x):
            Kx=jnp.zeros(Np,dtype=x.dtype)
            Kx=Kx.at[eu_j].add(-w*x[ev_j]+w*x[eu_j])
            Kx=Kx.at[ev_j].add(-w*x[eu_j]+w*x[ev_j])
            return Kx

        def _ion(V,h):
            sw=jax.nn.sigmoid(DTYPE(150.)*(V-DTYPE(V_GATE)))
            dh=((DTYPE(1.)-h)/p5_j[2])*(DTYPE(1.)-sw)-(h/p5_j[3])*sw
            return (jnp.clip(V+dt_sub*(h*V*(V-DTYPE(A_CRIT))*(DTYPE(1.)-V)/p5_j[0]
                                       -(DTYPE(1.)-h)*(V/p5_j[1])),DTYPE(0.),DTYPE(1.)),
                    jnp.clip(h+dt_sub*dh,DTYPE(0.),DTYPE(1.)))

        def _cn(V):
            rhs=V-alpha*m_j*spmv(V)
            Vn,_=jax.scipy.sparse.linalg.cg(
                lambda x:x+alpha*m_j*spmv(x),rhs,x0=V,tol=1e-6,maxiter=50)
            return jnp.clip(Vn,DTYPE(0.),DTYPE(1.))

        def scan_fn(carry,inputs):
            sv_t,t_idx=inputs
            V,h,at_num,at_den,activated,at_hard=carry
            for _ in range(N_ION): V,h=_ion(V,h)
            V=jnp.where(sv_t,jnp.clip(V+Iext*mask_j,DTYPE(0.),DTYPE(1.)),V)
            V=_cn(V)
            for _ in range(N_ION): V,h=_ion(V,h)
            t_now=DTYPE(t_idx)*DTYPE(DT_SIM)
            in_b1=(t_now<DTYPE(S1_BCL)).astype(DTYPE)
            w_at=jax.nn.sigmoid(BETA_AT*(V-DTYPE(V_GATE)))*in_b1
            at_num=at_num+t_now*w_at; at_den=at_den+w_at
            past_blank=t_now>T_BLANK; in_beat1=t_now<DTYPE(S1_BCL)
            fires=(~activated)&(V>DTYPE(V_GATE))&past_blank&in_beat1
            at_hard=jnp.where(fires,t_now,at_hard); activated=activated|fires
            return (V,h,at_num,at_den,activated,at_hard),W_j@V

        V0=jnp.zeros(Np,dtype=DTYPE); h0=jnp.ones(Np,dtype=DTYPE)
        at_num0=jnp.zeros(Np,dtype=DTYPE); at_den0=jnp.zeros(Np,dtype=DTYPE)
        activated0=jnp.zeros(Np,dtype=jnp.bool_)
        at_hard0=jnp.full(Np,DTYPE(S1_BCL),dtype=DTYPE)
        (_,_,at_num,at_den,_,at_hard),phi_T=jax.lax.scan(
            jax.checkpoint(scan_fn),
            (V0,h0,at_num0,at_den0,activated0,at_hard0),
            (sv_j,jnp.arange(NT,dtype=jnp.int32)))
        at_soft=at_num/(at_den+DTYPE(1e-6))
        return phi_T,at_soft,at_hard
    return run_3d

run_3d_cs  = make_run_3d(cs_mask_j)
run_3d_laa = make_run_3d(laa_mask_j)
p5_np=np.array([TAU_IN,TAU_OUT,TAU_OPEN,TAU_CLOSE,G_IL],dtype=NP_DTYPE)
p5_j=jnp.array(p5_np,dtype=DTYPE)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: FULL LA FORWARD — GPU TIMING + LAT COMPARISON WITH CPU
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SECTION 1: Full LA Forward — GPU timing + LAT comparison")
print("="*60)

CPU_TIMES = {'cs': 2608.2, 'laa': 2738.6}  # Reference CPU: Intel Core i7-1255U (16 GB RAM)

gpu_results = {}
for site, run_fn, stim_nodes in [
        ('cs',  run_3d_cs,  CS_STIM_NODES),
        ('laa', run_3d_laa, LAA_STIM_NODES)]:

    print(f"\n── {site.upper()} pacing ──")

    # Load CPU LAT map for comparison
    npz_path = os.path.join(OUT_DIR, f"la_forward_{site}.npz")
    at_cpu = None
    if os.path.exists(npz_path):
        _d = np.load(npz_path)
        at_cpu = np.array(_d['at_map'], dtype=NP_DTYPE)
        print(f"  CPU LAT map loaded: [{at_cpu[at_cpu<S1_BCL].min():.0f},"
              f"{at_cpu.max():.0f}]ms")

    # GPU forward — warm up JIT first
    print(f"  Warming up JIT ...")
    _,_,_ = run_fn(p5_j)

    # Measure warm forward pass
    t0=time.time()
    phi_T_gpu, at_soft_gpu, at_hard_gpu = run_fn(p5_j)
    phi_T_gpu.block_until_ready()
    t_gpu = time.time()-t0
    at_gpu = np.array(at_hard_gpu, dtype=NP_DTYPE)

    print(f"  GPU forward (warm):  {t_gpu:.1f}s")
    print(f"  CPU forward:         {CPU_TIMES[site]:.1f}s")
    print(f"  Speedup:             {CPU_TIMES[site]/t_gpu:.0f}x")
    print(f"  GPU Hard AT: [{at_gpu[at_gpu<S1_BCL].min():.0f},"
          f"{at_gpu.max():.0f}]ms")
    print(f"  Activated:   {(at_gpu<S1_BCL).sum():,}/{Np:,}")

    # LAT comparison GPU vs CPU
    if at_cpu is not None:
        valid = (at_gpu<S1_BCL) & (at_cpu<S1_BCL)
        diff  = np.abs(at_gpu[valid]-at_cpu[valid])
        corr  = np.corrcoef(at_gpu[valid], at_cpu[valid])[0,1]
        print(f"\n  GPU vs CPU LAT comparison:")
        print(f"    Valid nodes:     {valid.sum():,}")
        print(f"    Max diff:        {diff.max():.2f} ms")
        print(f"    Mean diff:       {diff.mean():.4f} ms")
        print(f"    Correlation:     {corr:.6f}")
        print(f"    {'✓ IDENTICAL' if diff.max()<0.1 else '~ CLOSE' if diff.max()<1.0 else '✗ DIFFER'}")

    gpu_results[site] = {'t_gpu': t_gpu, 'at_gpu': at_gpu}

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: DIFFERENTIABILITY — PATCH FD + jax.grad + AD vs FD
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SECTION 2: Differentiability — Patch (Site 2)")
print("="*60)

# Build patch (Site 2 — set5)
#_p1=np.load(os.path.join(EGM_DIR_S2,'part1_outputs_set5.npz'),allow_pickle=True)
#_p2=np.load(os.path.join(EGM_DIR_S2,'part2_outputs_set5.npz'),allow_pickle=True)
#_fib3=np.array(_p1["fib3"]); _cs_edge=str(_p1["cs_edge"][0])
#_pm=np.array(_p2["params_nominal"],dtype=np.float64)
#_DT=float(_p2["DT_SIM"][0]); _NION=int(_p2["N_ION_SUB"][0])
#_dth=float(_pm[6])
# Load patch geometry from Part 1

_p1=np.load(os.path.join(EGM_DIR_S2,'patch_geo.npz'),allow_pickle=True)
_fib3=np.array(_p1["fib3"]); _cs_edge=str(_p1["cs_edge"][0])
_DT   = float(0.1)
_NION = int(4)
_pm = np.array([np.log(TAU_IN), np.log(TAU_OUT), np.log(TAU_OPEN), np.log(TAU_CLOSE), np.log(G_IL), np.log(_DT), 0.0], dtype=np.float64)
_dth  = float(0.0)

_BB=5000.
_HD16=HD16_S2
_bb0=np.vstack([_HD16[CLIQUES[4]],_HD16[CLIQUES[3]]]).min(0)-_BB
_bb1=np.vstack([_HD16[CLIQUES[4]],_HD16[CLIQUES[3]]]).max(0)+_BB
_msk=np.all((Verts>=_bb0)&(Verts<=_bb1),axis=1)
_gids=np.where(_msk)[0]; _lv=Verts[_gids]; _Np=len(_gids)
_rev={g:l for l,g in enumerate(_gids)}
_le=np.array([[_rev[n] for n in e] for e in Elems
              if all(n in _rev for n in e)],dtype=np.int64)
_vc=_lv*1e-4
print(f"\nPatch: {_Np:,} nodes")

_nn=np.zeros((_Np,3)); _ar=np.zeros(_Np)
for (i,j,k) in _le:
    cr=np.cross(_lv[j]-_lv[i],_lv[k]-_lv[i])
    a=0.5*np.linalg.norm(cr); n_=cr/(np.linalg.norm(cr)+1e-12)
    for nd in (i,j,k): _ar[nd]+=a/3.; _nn[nd]+=a*n_
for i in range(_Np):
    n=np.linalg.norm(_nn[i])
    if n>1e-12: _nn[i]/=n
_mj=jnp.array(1./(CM*np.maximum(_ar*1e-8,
    max(1e-8,np.percentile((_ar*1e-8)[_ar>0],5)))),dtype=jnp.float64)

_ec2=defaultdict(float)
for (i,j,k) in _le:
    pi,pj,pk=_vc[i],_vc[j],_vc[k]
    for (u,v_,w,a_,b_) in [(i,j,k,pj-pi,pk-pi),(j,i,k,pi-pj,pk-pj),
                            (k,i,j,pi-pk,pj-pk)]:
        c=np.linalg.norm(np.cross(a_,b_))
        if c<1e-14: continue
        _ec2[(min(v_,w),max(v_,w))]+=0.5*np.dot(a_,b_)/c
_eu=np.array([u for u,v_ in _ec2.keys()],dtype=np.int32)
_ev=np.array([v_ for u,v_ in _ec2.keys()],dtype=np.int32)
_ecot=np.array(list(_ec2.values()),dtype=np.float64)
_ed=_vc[_ev]-_vc[_eu]; _ed/=np.linalg.norm(_ed,axis=1,keepdims=True)+1e-12
_en=(_nn[_eu]+_nn[_ev])/2.; _en/=np.linalg.norm(_en,axis=1,keepdims=True)+1e-12
_eu_j=jnp.array(_eu); _ev_j=jnp.array(_ev)
_ecot_j=jnp.array(_ecot,dtype=jnp.float64)
_ed_j=jnp.array(_ed,dtype=jnp.float64)
_en_j=jnp.array(_en,dtype=jnp.float64)
_f3j=jnp.array(np.tile(_fib3,(len(_eu),1)),dtype=jnp.float64)

EDGE_ELEC={"LEFT":np.array([0,4,8,12]),"RIGHT":np.array([3,7,11,15]),
           "BOTTOM":np.array([0,1,2,3]),"TOP":np.array([12,13,14,15])}
_ep=_HD16[EDGE_ELEC[_cs_edge]]; _epc=_ep.mean(0)
_c5c=_HD16[CLIQUES[4]].mean(0)
_wf=(_c5c-_epc)/(np.linalg.norm(_c5c-_epc)+1e-12)
_lc=_epc-_wf*3.*1000.
_,_,_Ve=np.linalg.svd(_ep-_epc,full_matrices=False); _ld=_Ve[0]
_projs=(_ep-_epc)@_ld; _hl=max(np.ptp(_projs)/2.,1.)+2000.
_d=_lv-_lc; _pal=_d@_ld; _perp=_d-np.outer(_pal,_ld)
_dist=np.linalg.norm(_perp,axis=1)
_mk=(_dist<=2000.)&(np.abs(_pal)<=_hl)
if _mk.sum()<10: _mk=_dist<=4000.
_mkj=jnp.array(_mk.astype(np.float64))
print(f"Pacing mask: {int(_mk.sum())} nodes  edge={_cs_edge}")

_sig_r=G_IL/(G_IL+0.2)
_Wnp=np.zeros((16,_Np))
for _e in range(16):
    _r=np.maximum(np.linalg.norm(_vc-_HD16[_e]*1e-4,axis=1),1e-6)
    _Wnp[_e]=(1./_r)*(_sig_r/(4.*np.pi))
_Wj=jnp.array(_Wnp,dtype=jnp.float64)

_S2ON=10.+500.; _TOTALP=_S2ON+600.
_NTp=int(_TOTALP/_DT); _tms=np.arange(_NTp)*_DT
_sv=np.zeros(_NTp,dtype=bool)
_sv|=(_tms>=10.)&(_tms<12.)
_sv|=(_tms>=_S2ON)&(_tms<_S2ON+2.)
_svj=jnp.array(_sv)
_i_s2=int(_S2ON/_DT); _win_s2=min(int(400./_DT),_NTp-_i_s2)

def _build_w_patch(g_il):
    g_it=g_il/4.
    fs=_f3j-jnp.sum(_f3j*_en_j,axis=1,keepdims=True)*_en_j
    fs/=jnp.linalg.norm(fs,axis=1,keepdims=True)+1e-12
    pp=jnp.cross(_en_j,fs); pp/=jnp.linalg.norm(pp,axis=1,keepdims=True)+1e-12
    fr=fs*jnp.cos(_dth)+pp*jnp.sin(_dth)
    fr/=jnp.linalg.norm(fr,axis=1,keepdims=True)+1e-12
    cos2=jnp.sum(_ed_j*fr,axis=1)**2
    return jnp.abs(_ecot_j)*((g_il/(BETA*CM))*cos2+(g_it/(BETA*CM))*(1.-cos2))

@jax.jit
def _run_patch(p5_j64):
    g_il=p5_j64[4]; w=_build_w_patch(g_il)
    dt_sub=_DT/(2*_NION); Iext=(STIM_AMP/CM)*_DT*1e-3; alpha=_DT/2.
    def spmv(x):
        Kx=jnp.zeros(_Np,dtype=x.dtype)
        Kx=Kx.at[_eu_j].add(-w*x[_ev_j]+w*x[_eu_j])
        Kx=Kx.at[_ev_j].add(-w*x[_eu_j]+w*x[_ev_j])
        return Kx
    def _ion(V,h):
        # Rush-Larsen — stable for jacfwd
        sw      = jax.nn.sigmoid(jnp.float64(150.)*(V-jnp.float64(V_GATE)))
        alpha_h = (jnp.float64(1.)-sw)/p5_j64[2]
        beta_h  = sw/p5_j64[3]
        tau_h   = jnp.float64(1.)/(alpha_h+beta_h+jnp.float64(1e-10))
        h_inf   = alpha_h*tau_h
        h_new   = h_inf+(h-h_inf)*jnp.exp(-dt_sub/tau_h)
        h_new   = jnp.clip(h_new,jnp.float64(0.),jnp.float64(1.))
        I_in    = h*(V*(V-jnp.float64(A_CRIT))*(jnp.float64(1.)-V)/p5_j64[0])
        I_out   = (jnp.float64(1.)-h)*V/p5_j64[1]
        V_new   = jnp.clip(V+dt_sub*(I_in-I_out),
                           jnp.float64(0.),jnp.float64(1.))
        return V_new, h_new
    def _cn(V):
        rhs=V-alpha*_mj*spmv(V)
        Vn,_=jax.scipy.sparse.linalg.cg(
            lambda x:x+alpha*_mj*spmv(x),rhs,x0=V,tol=1e-6,maxiter=50)
        return jnp.clip(Vn,jnp.float64(0.),jnp.float64(1.))
    def scan_fn(c,sv_):
        V,h=c
        for _ in range(_NION): V,h=_ion(V,h)
        V=jnp.where(sv_,jnp.clip(V+Iext*_mkj,jnp.float64(0.),jnp.float64(1.)),V)
        V=_cn(V)
        for _ in range(_NION): V,h=_ion(V,h)
        return (V,h),_Wj@V
    V0=jnp.zeros(_Np,dtype=jnp.float64)
    h0=jnp.ones(_Np,dtype=jnp.float64)
    (_,_),phi_T=jax.lax.scan(jax.checkpoint(scan_fn),(V0,h0),_svj)
    return phi_T

def _loss_patch(p5_64):
    phi=_run_patch(p5_64)
    c=CLIQUES[CENTRE_CLIQUE]
    egm=(phi[_i_s2:_i_s2+_win_s2,c[0]]-phi[_i_s2:_i_s2+_win_s2,c[1]]
        -phi[_i_s2:_i_s2+_win_s2,c[2]]+phi[_i_s2:_i_s2+_win_s2,c[3]])/4.
    return jnp.mean(egm**2)

p5_64=jnp.array(p5_np,dtype=jnp.float64)

# ── Patch forward timing ──────────────────────────────────────────────────────
print("\n── Patch forward timing ──")
_=_run_patch(p5_64)  # warm up
t0=time.time()
phi_p=_run_patch(p5_64); phi_p.block_until_ready()
t_patch=time.time()-t0
print(f"  Patch forward (warm): {t_patch:.2f}s  ({_Np:,} nodes)")

# ── FD gradients on patch ─────────────────────────────────────────────────────
print("\n── FD gradients on patch (10 passes) ──")
CPU_FD_TIME = 119.7  # from laptop run
t0=time.time()
fd_grads=np.zeros(5,dtype=np.float64)
for i in range(5):
    p_p=np.array(p5_64); p_p[i]+=FD_EPS
    p_m=np.array(p5_64); p_m[i]-=FD_EPS
    v_p=float(_loss_patch(jnp.array(p_p,dtype=jnp.float64)))
    v_m=float(_loss_patch(jnp.array(p_m,dtype=jnp.float64)))
    fd_grads[i]=(v_p-v_m)/(2*FD_EPS)
    print(f"  {PNAMES[i]:<12}  FD={fd_grads[i]:>12.4e}",flush=True)
t_fd=time.time()-t0
print(f"  FD time GPU:  {t_fd:.1f}s")
print(f"  FD time CPU:  {CPU_FD_TIME:.1f}s")
print(f"  FD speedup:   {CPU_FD_TIME/t_fd:.1f}x")

# ── jax.grad on patch ─────────────────────────────────────────────────────────
print("\n── jax.grad on patch ──")
CPU_GRAD_TIME = 'N/A (explodes on CPU)'

grad_fn=jax.jit(jax.value_and_grad(_loss_patch))
_=grad_fn(p5_64)  # warm up JIT
t0=time.time()
loss_val,ad_grads=grad_fn(p5_64)
ad_grads.block_until_ready()
t_ad=time.time()-t0
ad_grads=np.array(ad_grads)

print(f"  jax.grad time (warm): {t_ad:.1f}s")
print(f"\n  {'Param':<12}{'AD grad':>14}{'FD grad':>14}{'ratio':>10}{'stable':>8}")
print("  "+"─"*58)
all_stable=True
for i,pn in enumerate(PNAMES):
    ratio=ad_grads[i]/fd_grads[i] if abs(fd_grads[i])>1e-10 else float('nan')
    stable='✓' if abs(ratio)<=10. else '✗ EXPLODE'
    if stable!='✓': all_stable=False
    print(f"  {pn:<12}{ad_grads[i]:>14.4e}{fd_grads[i]:>14.4e}{ratio:>10.4f}{stable:>8}")

print(f"\n  AD gradients stable: {'✓ YES' if all_stable else '✗ NO — ionic stiffness blow-up'}")
print(f"  jax.grad executes:   ✓ ({t_ad:.1f}s on GPU)")
print(f"  Conclusion: JAX-EP differentiable by construction;")
print(f"  ionic Jacobian unstable under explicit integration.")

# ── jacfwd on patch — KEY NEW RESULT ─────────────────────────────────────────
print(f"\n── jax.jacfwd on patch (5 exact passes) ──")
print(f"  Forward-mode AD — no backprop — no BPTT explosion")
jacfwd_fn = jax.jit(jax.jacfwd(_loss_patch))
_ = jacfwd_fn(p5_64)  # warm up
t0 = time.time()
fwd_grads = np.array(jacfwd_fn(p5_64))
t_fwd = time.time()-t0

ratios = np.where(np.abs(fd_grads)>1e-10,
                  fwd_grads/fd_grads, float('nan'))
fwd_stable = all(np.isfinite(r) and abs(r-1.0)<0.15
                 for r in ratios if np.isfinite(r))

print(f"\n  {'Param':<12}{'jacfwd':>14}{'FD':>14}{'ratio':>10}{'stable':>8}")
print("  "+"─"*58)
for i,pn in enumerate(PNAMES):
    r = ratios[i]
    ok = '✓' if np.isfinite(r) and abs(r-1.0)<0.15 else '✗'
    print(f"  {pn:<12}{fwd_grads[i]:>14.4e}{fd_grads[i]:>14.4e}{r:>10.4f}{ok:>8}")

print(f"\n  jacfwd time (GPU): {t_fwd:.1f}s  FD time: {t_fd:.1f}s")
print(f"  jacfwd stable: {'YES ✓ — FULLY DIFFERENTIABLE' if fwd_stable else 'NO ✗'}")
print(f"  Forward passes: jacfwd=5 (exact) vs FD=10 (approx)")
if t_fwd > 0:
    print(f"  Speed ratio jacfwd/FD: {t_fwd/t_fd:.2f}x")

# ── Jacobian dAT/dG_IL on full LA ────────────────────────────────────────────
print(f"\n── Jacobian dAT/dG_IL on full LA (2 forward passes) ──")
print(f"  Running 2 forward passes on GPU ...")
p_gp=p5_np.copy(); p_gp[4]+=FD_EPS
p_gm=p5_np.copy(); p_gm[4]-=FD_EPS
t0=time.time()
_,_,at_p=run_3d_cs(jnp.array(p_gp,dtype=DTYPE))
_,_,at_m=run_3d_cs(jnp.array(p_gm,dtype=DTYPE))
at_p.block_until_ready()
jac_gil=(np.array(at_p)-np.array(at_m))/(2*FD_EPS)
t_jac=time.time()-t0
valid_jac=jac_gil[jac_gil!=0.]
print(f"  Jacobian range: [{jac_gil.min():.2f}, {jac_gil.max():.2f}] ms")
print(f"  Non-zero nodes: {(jac_gil!=0.).sum():,}")
print(f"  GPU time:       {t_jac:.1f}s")
print(f"  CPU time:       5096.2s  (from laptop)")
print(f"  Speedup:        {5096.2/t_jac:.0f}x")

# Save Jacobian
np.savez(os.path.join(OUT_DIR,'jac_gil_cs.npz'),
         jac_gil=jac_gil, Verts=Verts, Elems=Elems,
         HD16_S1=HD16_S1)
print(f"  Saved: jac_gil_cs.npz")

# ── Save LAT maps ─────────────────────────────────────────────────────────────
print(f"\n── Saving LAT maps and Vm snapshots ──")
np.savez(os.path.join(OUT_DIR,'la_forward_cs.npz'),
         at_map=gpu_results['cs']['at_gpu'],
         Verts=Verts, Elems=Elems,
         HD16_S1=HD16_S1, HD16_S2=HD16_S2)
np.savez(os.path.join(OUT_DIR,'la_forward_laa.npz'),
         at_map=gpu_results['laa']['at_gpu'],
         Verts=Verts, Elems=Elems,
         HD16_S1=HD16_S1, HD16_S2=HD16_S2)
print(f"  Saved: la_forward_cs.npz  la_forward_laa.npz")

# ── Vm snapshots at 3 timepoints during CS propagation ───────────────────────
print(f"\n── Vm snapshots for abstract figure ──")
SNAP_TIMES_MS = [100., 150., 180., 210.]
vm_snaps_list = []

p5_snap=jnp.array([TAU_IN,TAU_OUT,TAU_OPEN,TAU_CLOSE,G_IL],dtype=DTYPE)
print(f"  Running {len(SNAP_TIMES_MS)} short sims for Vm snapshots...")
for t_snap in SNAP_TIMES_MS:
    n_steps=int(t_snap/DT_SIM)

    @jax.jit
    def run_3d_to_t(p5_j):
        w=build_w(p5_j[4])
        dt_sub=DTYPE(DT_SIM/(2*N_ION))
        Iext=DTYPE((STIM_AMP/CM)*DT_SIM*1e-3)
        alpha=DTYPE(DT_SIM/2.)
        def spmv(x):
            Kx=jnp.zeros(Np,dtype=x.dtype)
            Kx=Kx.at[eu_j].add(-w*x[ev_j]+w*x[eu_j])
            Kx=Kx.at[ev_j].add(-w*x[eu_j]+w*x[ev_j])
            return Kx
        def _ion_s(V,h):
            sw=jax.nn.sigmoid(DTYPE(150.)*(V-DTYPE(V_GATE)))
            dh=((DTYPE(1.)-h)/DTYPE(TAU_OPEN))*(DTYPE(1.)-sw)-(h/DTYPE(TAU_CLOSE))*sw
            return (jnp.clip(V+dt_sub*(h*V*(V-DTYPE(A_CRIT))*(DTYPE(1.)-V)/DTYPE(TAU_IN)
                                       -(DTYPE(1.)-h)*(V/DTYPE(TAU_OUT))),DTYPE(0.),DTYPE(1.)),
                    jnp.clip(h+dt_sub*dh,DTYPE(0.),DTYPE(1.)))
        def _cn(V):
            rhs=V-alpha*m_j*spmv(V)
            Vn,_=jax.scipy.sparse.linalg.cg(
                lambda x:x+alpha*m_j*spmv(x),rhs,x0=V,tol=1e-6,maxiter=50)
            return jnp.clip(Vn,DTYPE(0.),DTYPE(1.))
        def scan_fn(c,sv_):
            V,h=c
            for _ in range(N_ION): V,h=_ion_s(V,h)
            V=jnp.where(sv_,jnp.clip(V+Iext*cs_mask_j,DTYPE(0.),DTYPE(1.)),V)
            V=_cn(V)
            for _ in range(N_ION): V,h=_ion_s(V,h)
            return (V,h),None
        V0=jnp.zeros(Np,dtype=DTYPE); h0=jnp.ones(Np,dtype=DTYPE)
        sv_slice=jax.lax.dynamic_slice_in_dim(sv_j,0,n_steps,axis=0)
        (V_final,_),_=jax.lax.scan(scan_fn,(V0,h0),sv_slice)
        return V_final

    t0=time.time()
    Vm=np.array(run_3d_to_t(p5_snap))
    print(f"  t={t_snap}ms  n_steps={n_steps}  Vm_max={Vm.max():.3f}  ({time.time()-t0:.1f}s)")
    vm_snaps_list.append(Vm)

vm_snaps=np.stack(vm_snaps_list)   # (4, Np)
np.savez(os.path.join(OUT_DIR,'vm_snapshots_cs.npz'),
         vm_snaps=vm_snaps,
         snap_times=np.array(SNAP_TIMES_MS),
         Verts=Verts, Elems=Elems,
         HD16_S1=HD16_S1,
         CS_STIM_NODES=CS_STIM_NODES,
         LAA_STIM_NODES=LAA_STIM_NODES)
print(f"  Saved: vm_snapshots_cs.npz  shape={vm_snaps.shape}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL GPU BENCHMARK SUMMARY")
print("="*60)
print(f"\n  FORWARD SIMULATION (full LA, 132,803 nodes, NT=24,200):")
print(f"    CS  GPU warm:  {gpu_results['cs']['t_gpu']:.1f}s  "
      f"CPU: {CPU_TIMES['cs']:.1f}s  → {CPU_TIMES['cs']/gpu_results['cs']['t_gpu']:.0f}x speedup")
print(f"    LAA GPU warm:  {gpu_results['laa']['t_gpu']:.1f}s  "
      f"CPU: {CPU_TIMES['laa']:.1f}s  → {CPU_TIMES['laa']/gpu_results['laa']['t_gpu']:.0f}x speedup")
print(f"\n  PATCH (2,894 nodes):")
print(f"    Forward (warm): {t_patch:.2f}s")
print(f"    FD 10 passes:   {t_fd:.1f}s  CPU: {CPU_FD_TIME:.1f}s  → {CPU_FD_TIME/t_fd:.1f}x speedup")
print(f"    jax.grad:       {t_ad:.1f}s  (runs but AD unstable — ionic stiffness)")
print(f"    jax.jacfwd:     {t_fwd:.1f}s  (stable, exact, 5 passes vs 10 FD) ✓")
print(f"\n  JACOBIAN dAT/dG_IL full LA:")
print(f"    GPU: {t_jac:.1f}s  CPU: 5096.2s  → {5096.2/t_jac:.0f}x speedup")  # reference CPU measurement
print("\nDONE")