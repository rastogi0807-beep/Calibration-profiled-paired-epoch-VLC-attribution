"""Reduced, synchronized small-signal VLC transfer twin. All outputs are synthetic."""
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]
F=np.array([0,.5,1,2,3,4,6,8,12,18,26,38.])
LOW=np.array([np.log(2),np.log(2),-3.]);HIGH=np.array([np.log(40),np.log(60),3.])
NAMES=['NoChange','Emitter','Channel','Mixed']
METHODS=['LinkMulti','RefCold','RefAnalytic','RefLearned','RefNetworkOnly']
CASES=['Nominal','LowSNR','RefGainShift','RefPoleError','DiffuseDelay']
CFG=dict(probes=F.tolist(),reference_pole_MHz=60.,sigma_nominal=.012,sigma_low=.045,bic_margin=2.,residual_quantile=.99,primary_episodes=256,training_examples=24000,validation_examples=4000,units='MHz bandwidths; ns delay; normalized electrical transfer gains')
def transfer(x,f=F,refpole=60.,delay_ns=0.,derivative=False):
    b,p,k=np.exp(x[0]),np.exp(x[1]),expit(x[2]);e=1/(1+1j*f/b);d=1/(1+1j*f/p);lag=np.exp(-2j*np.pi*f*delay_ns*.001);c=(1-k)+k*d*lag;ref=1/(1+1j*f/refpole);hl=e*c;hr=e*ref
    if not derivative:return hl,hr
    jl=np.stack([e*(1-e)*c,e*k*d*(1-d)*lag,e*k*(1-k)*(d*lag-1)],axis=-1);jr=np.stack([e*(1-e)*ref,np.zeros(len(f),complex),np.zeros(len(f),complex)],axis=-1)
    return hl,hr,jl,jr
def swap(x):
    b,p,k=np.exp(x[0]),np.exp(x[1]),expit(x[2]);kk=1-b*(1-k)/p
    return np.array([np.log(p),np.log(b),logit(kk)]) if 0<kk<1 else None
def profile(y,h,j=None):
    den=float(np.vdot(h,h).real);num=float(np.vdot(h,y).real);gain=max(1e-8,num/den);res=gain*h-y
    if j is None:return res,gain
    dg=(np.real(np.conj(j).T@y)*den-num*2*np.real(np.conj(h)@j))/den**2
    if gain==1e-8:dg*=0
    jac=h[:,None]*dg[None,:]+gain*j
    return res,gain,jac
def residual(x,yl,yr,sigmas,use_ref=True,f=F):
    hl,hr,jl,jr=transfer(x,f,derivative=True);rl,gl,dl=profile(yl,hl,jl);parts=[rl*np.sqrt(2)/sigmas[0]];js=[dl*np.sqrt(2)/sigmas[0]];gains=[gl]
    if use_ref:
        rr,gr,dr=profile(yr,hr,jr);parts.append(rr*np.sqrt(2)/sigmas[1]);js.append(dr*np.sqrt(2)/sigmas[1]);gains.append(gr)
    z=np.concatenate(parts);jj=np.concatenate(js,axis=0)
    return np.r_[z.real,z.imag],np.r_[jj.real,jj.imag],gains
def features(yl,yr,sigmas):
    yl=np.asarray(yl);yr=np.asarray(yr);sl,sr=np.asarray(sigmas)[...,0],np.asarray(sigmas)[...,1];dl=np.maximum(abs(yl[...,0]),1e-4);dr=np.maximum(abs(yr[...,0]),1e-4);l=yl/dl[...,None];r=yr/dr[...,None]
    return np.concatenate([l.real,l.imag,r.real,r.imag,(sl/dl)[...,None],(sr/dr)[...,None]],axis=-1)
def observation(x,gains,seed,sigmas=(.012,.012),refpole=60.,delay_ns=0.,f=F):
    rng=np.random.default_rng(seed);hl,hr=transfer(x,f,refpole,delay_ns);z=(rng.normal(size=(2,len(f)))+1j*rng.normal(size=(2,len(f))))/np.sqrt(2);return gains[0]*hl+sigmas[0]*z[0],gains[1]*hr+sigmas[1]*z[1]
def episode(seed):
    r=np.random.default_rng(seed);base=np.array([r.uniform(np.log(6),np.log(20)),r.uniform(np.log(8),np.log(28)),logit(r.uniform(.2,.65))]);cls=(seed-10000)%4;x=base.copy();g=r.uniform(.6,1.4,2)
    if cls in [1,3]:x[0]+=np.log(r.uniform(.55,.8))
    if cls in [2,3]:x[1]+=np.log(r.uniform(.5,.75));x[2]=logit(min(.9,expit(base[2])+r.uniform(.1,.2)))
    return dict(seed=seed,baseline=base,truth=x,label=cls,gains=g)
def modified_observation(ep,case):
    sig=.045 if case=='LowSNR' else .012;g=ep['gains'].copy();rp=60.;delay=0.
    if case=='RefGainShift':g[1]*=.6
    if case=='RefPoleError':rp=20.
    if case=='DiffuseDelay':delay=6.
    yl,yr=observation(ep['truth'],g,ep['seed']+700000,(sig,sig),rp,delay);return yl,yr,np.array([sig,sig])
