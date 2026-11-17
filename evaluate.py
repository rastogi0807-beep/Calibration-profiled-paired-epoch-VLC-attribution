from model import *
from inference import infer
import pandas as pd,joblib,json,time,platform

def result_row(ep,case,method,out,yl,yr,sigmas):
    x=out['x'];truth=ep['truth'];base=ep['baseline'];ref=method!='LinkMulti';hl,hr=transfer(x);true_hl,true_hr=transfer(truth,refpole=20 if case=='RefPoleError' else 60,delay_ns=6 if case=='DiffuseDelay' else 0);fit_hl=out['gains'][0]*hl;true_link=ep['gains'][0]*true_hl;err=x-truth;in_scope=case not in ['RefPoleError','DiffuseDelay'];hit=out['label']==ep['label'];se=out['standard_error'];ci_issued=out['rank']==3 and method!='RefNetworkOnly';covered=(np.abs(err)<=1.95996398454*se) if ci_issued else np.array([np.nan]*3)
    row=dict(seed=ep['seed'],case=case,method=method,in_scope=in_scope,true_label=ep['label'],predicted_label=out['label'],accepted=out['accepted'],correct=bool(hit),accepted_correct=bool(out['accepted'] and hit),margin=out['margin'],residual_ok=out['residual_ok'],rss=out['rss'],full_rss=out['full_rss'],calls=out['calls'],seconds=out['seconds'],link_nmse=float(np.mean(abs(fit_hl-true_link)**2)/np.mean(abs(true_link)**2)),condition=out['condition'],rank=out['rank'],ci_issued=ci_issued,converged=out['converged'])
    for j,name in enumerate(['log_b','log_p','logit_k']):row.update({name+'_estimate':float(x[j]),name+'_truth':float(truth[j]),name+'_baseline':float(base[j]),name+'_error':float(err[j]),name+'_se':float(se[j]),name+'_covered':float(covered[j])})
    for j in range(4):row['bic'+str(j)]=float(out['bic'][j])
    return row

def summarize(df,path):
    rows=[]
    for (case,method),p in df.groupby(['case','method'],sort=False):
        selected=p[p.accepted];row=dict(case=case,method=method,n=len(p),in_scope=bool(p.in_scope.iloc[0]),hard_accuracy=p.correct.mean(),coverage=p.accepted.mean(),accepted_correct=p.accepted_correct.mean(),accepted_accuracy=selected.correct.mean() if len(selected) else np.nan,residual_pass=p.residual_ok.mean(),calls=p.calls.mean(),median_ms=p.seconds.median()*1000,link_nmse=p.link_nmse.mean(),ci_issued=p.ci_issued.mean(),converged=p.converged.mean())
        for name in ['log_b','log_p','logit_k']:row[name+'_rmse']=float(np.sqrt(np.mean(p[name+'_error']**2)));row[name+'_coverage']=p[name+'_covered'].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False)

def main():
    st=time.time();bundle=joblib.load(ROOT/'data/initializer.joblib');rows=[];observations=[]
    for seed in range(10000,10256):
        ep=episode(seed)
        for case in CASES:
            yl,yr,sigmas=modified_observation(ep,case);observations.append(dict(seed=seed,case=case,baseline=ep['baseline'].tolist(),truth=ep['truth'].tolist(),gains=ep['gains'].tolist(),label=ep['label'],sigmas=sigmas.tolist(),yl_re=yl.real.tolist(),yl_im=yl.imag.tolist(),yr_re=yr.real.tolist(),yr_im=yr.imag.tolist()))
            for method in METHODS:
                out=infer(method,ep['baseline'],yl,yr,sigmas,bundle);rows.append(result_row(ep,case,method,out,yl,yr,sigmas))
        if (seed-9999)%32==0:print('primary episodes',seed-9999,flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'data/primary_results.csv',index=False);summarize(df,ROOT/'data/primary_summary.csv');(ROOT/'data/primary_observations.json').write_text(json.dumps(observations))
    intervals=[];diffs=[];rng=np.random.default_rng(9300)
    for case in CASES[:3]:
        p=df[df['case']==case];ix=rng.integers(0,256,(4000,256));samples={}
        for method in METHODS:
            one=p[p.method==method].sort_values('seed');samples[method]={}
            for metric in ['correct','accepted','accepted_correct']:
                v=one[metric].to_numpy(dtype=float);boot=v[ix].mean(1);lo,hi=np.quantile(boot,[.025,.975]);intervals.append(dict(case=case,method=method,metric=metric,mean=float(v.mean()),low=float(lo),high=float(hi)));samples[method][metric]=boot
        for method in METHODS:
            if method=='RefLearned':continue
            for metric in ['correct','accepted_correct']:
                d=samples['RefLearned'][metric]-samples[method][metric];lo,hi=np.quantile(d,[.025,.975]);diffs.append(dict(case=case,comparison='RefLearned_vs_'+method,metric=metric,mean=float(p[p.method=='RefLearned'][metric].mean()-p[p.method==method][metric].mean()),low=float(lo),high=float(hi)))
    pd.DataFrame(intervals).to_csv(ROOT/'data/bootstrap_intervals.csv',index=False);pd.DataFrame(diffs).to_csv(ROOT/'data/paired_differences.csv',index=False)
    paired=[];paired_obs=[];maximum_identity=0.
    for seed in range(20000,20128):
        rng=np.random.default_rng(seed);q=rng.uniform(8,20);s=rng.uniform(.5,.8);kk=rng.uniform(.25,.6);base=np.array([np.log(q),np.log(q),logit(kk)]);states=[np.array([np.log(q*s),np.log(q),logit(kk)]),np.array([np.log(q),np.log(q*s),logit(1-s*(1-kk))])];g=rng.uniform(.6,1.4,2);obs=[observation(x,g,seed+900000,(.012,.012)) for x in states];common=obs[0][0];maximum_identity=max(maximum_identity,float(np.max(abs(obs[0][0]-obs[1][0]))))
        for j,truth in enumerate(states):
            yl=common.copy();yr=observation(truth,g,seed+910000+j,(.012,.012))[1];ep=dict(seed=seed,baseline=base,truth=truth,label=j+1,gains=g);paired_obs.append(dict(pair_seed=seed,label=j+1,baseline=base.tolist(),truth=truth.tolist(),gains=g.tolist(),yl_re=yl.real.tolist(),yl_im=yl.imag.tolist(),yr_re=yr.real.tolist(),yr_im=yr.imag.tolist()))
            for method in METHODS:
                out=infer(method,base,yl,yr,np.array([.012,.012]),bundle);paired.append(result_row(ep,'ExchangePairs',method,out,yl,yr,[.012,.012]))
        if (seed-19999)%32==0:print('ambiguity pairs',seed-19999,flush=True)
    pd.DataFrame(paired).to_csv(ROOT/'data/exchange_results.csv',index=False);summarize(pd.DataFrame(paired),ROOT/'data/exchange_summary.csv');(ROOT/'data/exchange_observations.json').write_text(json.dumps(paired_obs));(ROOT/'data/execution_report.json').write_text(json.dumps(dict(config=CFG,primary_rows=len(df),observation_records=len(observations),exchange_rows=len(paired),paired_contexts=128,paired_link_identity_error=maximum_identity,elapsed_seconds=time.time()-st,python=platform.python_version(),numpy=np.__version__,scope='Synthetic small-signal model identification; no hardware or causal guarantee beyond the stated hypothesis model.'),indent=2));print('Complete',time.time()-st,flush=True)
if __name__=='__main__':main()
