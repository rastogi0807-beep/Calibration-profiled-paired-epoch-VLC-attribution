from model import *
from scipy.special import logit
import pandas as pd,json,platform,subprocess
checks={};rng=np.random.default_rng(4500);errors=[]
for i in range(12):
 x=np.array([rng.uniform(np.log(4),np.log(25)),rng.uniform(np.log(5),np.log(35)),logit(rng.uniform(.1,.85))]);yl,yr=observation(x,[.8,1.1],4500+i,(.012,.02))
 r,j,_=residual(x,yl,yr,(.012,.02),True)
 num=np.column_stack([(residual(x+np.eye(3)[k]*1e-5,yl,yr,(.012,.02),True)[0]-residual(x-np.eye(3)[k]*1e-5,yl,yr,(.012,.02),True)[0])/2e-5 for k in range(3)])
 errors.append(float(np.max(abs(j-num))))
checks['maximum_absolute_jacobian_difference']=max(errors);assert max(errors)<1e-6
p=pd.read_csv(ROOT/'data/primary_results.csv');checks['primary_rows']=len(p);assert len(p)==6400
checks['finite_parameter_estimates']=bool(np.isfinite(p[['log_b_estimate','log_p_estimate','logit_k_estimate']]).all().all())
checks['per_method_nominal_rows']=p[p['case']=='Nominal'].groupby('method').size().to_dict()
sub=p[p['case']=='Nominal'].pivot(index='seed',columns='method',values='correct');checks['refined_decisions_identical']=bool((sub['RefCold']==sub['RefLearned']).all() and (sub['RefAnalytic']==sub['RefLearned']).all());assert checks['refined_decisions_identical']
records=json.loads((ROOT/'data/primary_observations.json').read_text());checks['saved_observation_count']=len(records)
ex=pd.read_csv(ROOT/'data/exchange_results.csv');checks['exchange_independent_pairs']=ex.seed.nunique();checks['link_pair_accuracy_bound']=float(ex[ex.method=='LinkMulti'].correct.mean());assert checks['link_pair_accuracy_bound']<=.5+1e-12
checks['scipy_version']=__import__('scipy').__version__;checks['sklearn_version']=__import__('sklearn').__version__;checks['python']=platform.python_version();checks['numpy']=np.__version__
cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
checks['cpu']=[line for line in cpuinfo.splitlines() if line.startswith(('model name','cpu MHz'))][:2] or ['Virtual host does not expose CPU model']
(ROOT/'data/audit_report.json').write_text(json.dumps(checks,indent=2));print(json.dumps({k:v for k,v in checks.items() if k!='cpu'},indent=2))
