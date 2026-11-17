from model import *
from train import predict
from scipy.optimize import least_squares,minimize_scalar
from scipy.stats import chi2
import time
FREE=[[],[0],[1,2],[0,1,2]]
class Evaluator:
    def __init__(self,baseline,free,yl,yr,sigmas,reference=True,f=F):
        self.baseline=baseline.copy();self.free=np.array(free,int);self.yl=yl;self.yr=yr;self.sigmas=sigmas;self.reference=reference;self.f=f;self.calls=0;self.last=None
    def evaluate(self,z):
        if self.last is None or not np.array_equal(z,self.last):
            x=self.baseline.copy();x[self.free]=z;r,j,g=residual(x,self.yl,self.yr,self.sigmas,self.reference,self.f);self.value=(r,j[:,self.free],g,x);self.last=np.array(z).copy();self.calls+=1
        return self.value
    def fun(self,z):return self.evaluate(z)[0]
    def jac(self,z):return self.evaluate(z)[1]
def fit_one(base,free,start,yl,yr,sigmas,ref=True,f=F):
    e=Evaluator(base,free,yl,yr,sigmas,ref,f)
    if free:
        z=np.clip(start[free],LOW[free]+1e-7,HIGH[free]-1e-7);ans=least_squares(e.fun,z,jac=e.jac,bounds=(LOW[free],HIGH[free]),max_nfev=90,ftol=1e-9,xtol=1e-9,gtol=1e-9);r,j,g,x=e.evaluate(ans.x);success=bool(ans.success)
    else:r,j,g,x=e.evaluate(np.empty(0));success=True
    return dict(x=x,rss=float(r@r),jac=j,gains=g,calls=e.calls,success=success)
def analytic_start(base,yl,yr,sigmas,f=F):
    counter=[0]
    def score(t):
        counter[0]+=1;xx=base.copy();xx[0]=t;hr=transfer(xx,f)[1];rr=profile(yr,hr)[0];return float(np.vdot(rr,rr).real/sigmas[1]**2)
    a=minimize_scalar(score,method='bounded',bounds=(LOW[0],HIGH[0]),options=dict(xatol=1e-7));xx=base.copy();xx[0]=a.x;out=fit_one(xx,[1,2],xx,yl,yr,sigmas,False,f);counter[0]+=out['calls'];return out['x'],counter[0]
def infer(method,base,yl,yr,sigmas,bundle,f=F):
    st=time.perf_counter();ref=method!='LinkMulti';calls=0
    if method in ['RefLearned','RefNetworkOnly']:start=predict(bundle,yl,yr,sigmas)[0];starts=[base,start]
    elif method=='RefAnalytic':start,calls=analytic_start(base,yl,yr,sigmas,f);starts=[base,start]
    else:starts=[base,np.array([np.log(5),np.log(9),logit(.35)]),np.array([np.log(20),np.log(24),logit(.7)])]
    fits=[]
    for free in FREE:
        if method=='RefNetworkOnly':
            x=base.copy();x[free]=start[free];r,j,g=residual(x,yl,yr,sigmas,True,f);one=dict(x=x,rss=float(r@r),jac=j[:,free],gains=g,calls=1,success=True);calls+=1
        else:
            candidates=[fit_one(base,free,s,yl,yr,sigmas,ref,f) for s in (starts if free else [base])];calls+=sum(v['calls'] for v in candidates);one=min(candidates,key=lambda d:d['rss'])
        fits.append(one)
    dim=2*len(f)*(2 if ref else 1);ng=2 if ref else 1;bic=np.array([fit['rss']+(ng+len(fr))*np.log(dim) for fit,fr in zip(fits,FREE)]);order=np.argsort(bic,kind='stable');selected=int(order[0]);margin=float(bic[order[1]]-bic[selected]);dof=dim-ng-len(FREE[selected]);residual_ok=fits[selected]['rss']<=chi2.ppf(CFG['residual_quantile'],dof);accepted=bool(margin>=CFG['bic_margin'] and residual_ok)
    full=fits[-1];info=full['jac'].T@full['jac'];sv=np.linalg.svd(info,compute_uv=False);rank=int(np.sum(sv>sv[0]*1e-12));cov=np.linalg.pinv(info,rcond=1e-12);se=np.sqrt(np.maximum(0,np.diag(cov)));condition=float(sv[0]/max(sv[-1],1e-20));se=np.where(rank==3,se,np.nan)
    return dict(x=full['x'],selected_x=fits[selected]['x'],label=selected,accepted=accepted,margin=margin,residual_ok=bool(residual_ok),rss=fits[selected]['rss'],full_rss=full['rss'],bic=bic,standard_error=se,condition=condition,rank=rank,calls=calls,seconds=time.perf_counter()-st,gains=full['gains'],converged=all(v['success'] for v in fits))
