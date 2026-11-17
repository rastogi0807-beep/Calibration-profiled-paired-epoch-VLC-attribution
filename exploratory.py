from model import *
from train import predict
from inference import infer,fit_one
from scipy.special import logit
import joblib,pandas as pd,time,json

METHODS=['LinkMulti','RefCold','RefAnalytic','RefLearned','RefNetworkOnly']
bundle=joblib.load(ROOT/'data/initializer.joblib')
rows=[]
for sr in [.004,.012,.04,.12]:
 for seed in range(30000,30064):
  ep=episode(seed);base,true,label,gains=ep["baseline"],ep["truth"],ep["label"],ep["gains"];yl,yr=observation(true,gains,seed+700000,(.012,sr))
  for m in METHODS[1:]:
   a=infer(m,base,yl,yr,(.012,sr),bundle)
   rows.append(dict(reference_sigma=sr,seed=seed,method=m,correct=a['label']==label,accepted=a['accepted'],b_error=a['x'][0]-true[0],p_error=a['x'][1]-true[1],calls=a['calls']))
pd.DataFrame(rows).to_csv(ROOT/'data/exploratory_reference_noise.csv',index=False)
print('Reference-noise sweep complete',flush=True)

rows=[]
for k in [.02,.05,.1,.3,.6,.9]:
 for seed in range(31000,31064):
  ep=episode(seed);base,gains=ep["baseline"],ep["gains"];true=base.copy();true[2]=logit(k);yl,yr=observation(true,gains,seed+700000,(.012,.012))
  start=predict(bundle,yl,yr,(.012,.012))[0]
  ans=min([fit_one(base,[0,1,2],s,yl,yr,(.012,.012),True) for s in [base,start]],key=lambda a:a['rss'])
  info=ans['jac'].T@ans['jac'];sv=np.linalg.svd(info,compute_uv=False);rank=int(sum(sv>sv[0]*1e-12));se=np.sqrt(np.maximum(0,np.diag(np.linalg.pinv(info,rcond=1e-12))))
  error=ans['x']-true
  rows.append(dict(kappa=k,seed=seed,p_error=error[1],p_se=se[1],p_covered=bool(abs(error[1])<=1.96*se[1]) if rank==3 else False,rank=rank,kappa_est=float(expit(ans['x'][2])),at_bound=bool(np.any(abs(ans['x']-LOW)<1e-5) or np.any(abs(ans['x']-HIGH)<1e-5)),outside_fit_bounds=bool(true[2]<LOW[2])))
pd.DataFrame(rows).to_csv(ROOT/'data/exploratory_weak_diffuse.csv',index=False)

rows=[];x=np.array([np.log(12),np.log(18),logit(.4)])
for fm in [4,8,16,32,48]:
 f=F*fm/38;yl,yr=transfer(x,f)
 for ref in [False,True]:
  r,j,g=residual(x,yl,yr,(.012,.012),ref,f);info=j.T@j;sv=np.linalg.svd(info,compute_uv=False);se=np.sqrt(np.diag(np.linalg.pinv(info,rcond=1e-12)))
  rows.append(dict(maximum_frequency_MHz=fm,reference=ref,minimum_eigenvalue=sv[-1],condition=sv[0]/sv[-1],log_b_se=se[0],log_p_se=se[1],logit_k_se=se[2]))
pd.DataFrame(rows).to_csv(ROOT/'data/exploratory_fisher_span.csv',index=False)
print('Diffuse and Fisher sweeps complete',flush=True)

ep=episode(31999);base,true,gains=ep["baseline"],ep["truth"],ep["gains"];yl,yr=observation(true,gains,731999,(.012,.012))
for m in METHODS:infer(m,base,yl,yr,(.012,.012),bundle)
rows=[]
for si,seed in enumerate(range(32000,32032)):
 ep=episode(seed);base,true,gains=ep["baseline"],ep["truth"],ep["gains"];yl,yr=observation(true,gains,seed+700000,(.012,.012))
 for repeat in range(3):
  order=np.roll(METHODS,(si+repeat)%len(METHODS))
  for m in order:
   a=infer(m,base,yl,yr,(.012,.012),bundle)
   rows.append(dict(seed=seed,repeat=repeat,method=m,milliseconds=1000*a['seconds'],calls=a['calls']))
pd.DataFrame(rows).to_csv(ROOT/'data/exploratory_warmed_timing.csv',index=False)
print('Timing complete',flush=True)

p=pd.read_csv(ROOT/'data/primary_results.csv');rows=[]
for m,g in p[p['case']=='Nominal'].groupby('method',sort=False):
 for margin in [0,.5,1,2,4,8,16]:
  accept=(g['margin']>=margin)&g['residual_ok']
  rows.append(dict(method=m,minimum_margin=margin,coverage=float(accept.mean()),risk=float((~g.loc[accept,'correct']).mean()) if accept.any() else None))
pd.DataFrame(rows).to_csv(ROOT/'data/exploratory_risk_coverage.csv',index=False)

ex=pd.read_csv(ROOT/'data/exchange_results.csv');rng=np.random.default_rng(9400);rows=[]
# Each seed identifies a complete pair with a shared link observation.
for m,g in ex.groupby('method',sort=False):
 v=g.groupby('seed')[['correct','accepted']].mean().to_numpy();sel=rng.integers(0,len(v),(4000,len(v)))
 for j,metric in enumerate(['correct','accepted']):
  b=v[sel,j].mean(axis=1);rows.append(dict(method=m,metric=metric,mean=float(v[:,j].mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),independent_pairs=len(v)))
pd.DataFrame(rows).to_csv(ROOT/'data/exchange_cluster_intervals.csv',index=False)
(ROOT/'data/exploratory_report.json').write_text(json.dumps(dict(protocol='notes/exploratory_protocol.md',reference_noise_episodes_per_point=64,diffuse_episodes_per_point=64,timing_episodes=32,timing_repetitions=3,training_and_thresholds_unchanged=True),indent=2))
