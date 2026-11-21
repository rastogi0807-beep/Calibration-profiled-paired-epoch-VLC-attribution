"""Run with Python, NumPy and SciPy. Writes per-case results and summary."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import chi2

F = np.array([0,.5,1,2,3,4,6,8,12,18,26,38])
J = 1j * F
LOW = np.array([np.log(2), np.log(2), -3])
HIGH = np.array([np.log(40), np.log(60), 3])
SD = np.array([.08,.12,.30,.12])
LO = np.r_[LOW, np.log(45)]
HI = np.r_[HIGH, np.log(78)]
FREE = [[], [0], [1,2], [0,1,2]]
SCALE = np.sqrt(2)/.012
CONDITIONS = ['matched','two reflections','reference drift']
METHODS = ['joint','sequential']

def simulate(index, seed, condition):
    rng = np.random.default_rng(seed + index)
    label = index % 4
    x = np.r_[rng.uniform(np.log([6,8]),np.log([20,28])), logit(rng.uniform(.2,.65))]
    event = x.copy()
    if label in (1,3):
        event[0] += np.log(rng.uniform(.55,.80))
    if label in (2,3):
        event[1] += np.log(rng.uniform(.50,.75))
        event[2] = logit(expit(x[2])+rng.uniform(.10,.20))
    prior = np.clip(np.r_[x,np.log(60)] + rng.normal(size=4)*SD*np.array([1,1,1,0]), LO, HI)
    gains = rng.uniform(.6,1.4,(2,2))
    noise = .012/np.sqrt(2)*(rng.normal(size=(2,2,12))+1j*rng.normal(size=(2,2,12)))
    spectra = []
    for epoch, state in enumerate((x,event)):
        b,p = np.exp(state[:2])
        k = expit(state[2])
        emitter = b/(b+1j*F)
        refpole = 63 if condition == 'reference drift' and epoch else 60
        if condition == 'two reflections':
            diffuse = sum(w*np.exp(-2j*np.pi*F*t/1000)/(1+1j*F/(m*p))
                          for w,t,m in ((.63,3.5,1.3),(.37,11,.64)))
        else:
            diffuse = p/(p+1j*F)
        spectra.append([emitter*(1-k+k*diffuse), emitter/(1+1j*F/refpole)])
    return gains[:,:,None]*np.array(spectra)+noise, prior, label

def residual(y,x,r,tau=0):
    b,p = np.exp(x[:2])
    k = expit(x[2])
    e = b/(b+J)
    h = np.stack([e*(1-k+k*p/(p+J)*np.exp(-2*np.pi*J*tau/1000)),e*r/(r+J)])
    gain = np.maximum(1e-8,(h.conj()*y).real.sum(axis=1)/(abs(h)**2).sum(axis=1))
    error = (gain[:,None]*h-y)*SCALE
    return np.r_[error.real.ravel(),error.imag.ravel()]

def estimate(y,prior,mode):
    def baseline(v):
        return np.r_[residual(y[0],v[:3],np.exp(v[3])),(v-prior)/SD]
    def fit(fn,start,lower,upper):
        return least_squares(fn,start,bounds=(lower,upper),max_nfev=120,
                             ftol=1e-8,xtol=1e-8,gtol=1e-8)
    base = fit(baseline,prior,LO,HI)
    def fit_h(h,delay=False,start=None):
        ix = FREE[h]
        joint = mode == 'joint'
        def pieces(v):
            bv = v[:4] if joint else base.x
            ev = bv[:3].copy()
            ev[ix] = v[(4 if joint else 0):(4 if joint else 0)+len(ix)]
            return residual(y[0],bv[:3],np.exp(bv[3])), residual(y[1],ev,np.exp(bv[3]),v[-1] if delay else 0), (bv-prior)/SD
        def objective(v):
            return np.concatenate(pieces(v))
        x0 = np.r_[base.x if joint else [],base.x[ix]] if start is None else start
        lower = np.r_[LO if joint else [],LOW[ix]]
        upper = np.r_[HI if joint else [],HIGH[ix]]
        if delay:
            x0, lower, upper = np.r_[x0,0], np.r_[lower,0], np.r_[upper,20]
        v = fit(objective,x0,lower,upper).x if len(x0) else x0
        parts = pieces(v)
        q = sum(a@a for a in parts)
        raw = sum(a@a for a in parts[:2])
        return q,raw,v
    fits = [fit_h(h) for h in range(4)]
    scores = [v[0]+(8+len(FREE[h]))*np.log(96) for h,v in enumerate(fits)]
    h,second = np.argsort(scores)[:2]
    q,raw,v = fits[h]
    accept = scores[second]-scores[h] >= 2 and raw <= chi2.ppf(.99,88-len(FREE[h]))
    statistic = max(0.,q-fit_h(h,True,v)[0]) if accept else None
    return int(h),bool(accept),statistic

def wilson(n,N):
    z=1.959964
    p=n/N
    c=(p+z*z/(2*N))/(1+z*z/N)
    w=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/(1+z*z/N)
    return [round(100*(c-w),2),round(100*(c+w),2)]

def run():
    thresholds={}
    for mode in METHODS:
        scores=[estimate(*simulate(i,921000,'matched')[:2],mode)[2] for i in range(128)]
        passed=[x for x in scores if x is not None]
        thresholds[mode]=[float(np.quantile(passed,.95,method='linear')),len(passed)]
    results=[]
    summary={}
    rng=np.random.default_rng(921999)
    for condition in CONDITIONS:
        summary[condition]={}
        for mode in METHODS:
            rows=[]
            for i in range(96):
                y,prior,label=simulate(i,922000,condition)
                pred,a,stat=estimate(y,prior,mode)
                rows.append(dict(condition=condition,method=mode,case=i,label=label,prediction=pred,
                                 initial=a,statistic=stat,screened=a and stat<=thresholds[mode][0]))
            results.extend(rows)
            counts=[]
            for stage in ('initial','screened'):
                correct=sum(r[stage] and r['label']==r['prediction'] for r in rows)
                wrong=sum(r[stage] and r['label']!=r['prediction'] for r in rows)
                counts.append([int(correct),int(wrong),wilson(correct,96),wilson(wrong,96)])
            summary[condition][mode]=counts
        both=results[-192:]
        ds=[]
        for stage in ('initial','screened'):
            s=np.array([int(r[stage] and r['label']==r['prediction']) for r in both])
            diff=s[:96]-s[96:]
            indices=rng.integers(0,24,size=(4000,24,4))*4+np.arange(4)
            ds.append([int(diff.sum()),np.round(np.quantile(diff[indices].mean(axis=(1,2)),[.025,.975])*100,2).tolist()])
        summary[condition]['difference']=ds
    out=Path(__file__).resolve().parent
    with open(out/'cppe_paired_benchmark_cases.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    payload=dict(thresholds=thresholds,summary=summary)
    (out/'cppe_paired_benchmark_summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload))
if __name__=='__main__':
    run()
