"""Validate frozen records by default; optionally rerun the full simulations."""
from pathlib import Path
import argparse,subprocess,sys,json
import numpy as np,pandas as pd
from model import ROOT,episode,modified_observation
parser=argparse.ArgumentParser()
parser.add_argument('--rerun',action='store_true',help='Regenerate training and evaluation records; overwrites data and timings')
parser.add_argument('--build-documents',action='store_true',help='Rebuild Word documents; requires pandoc and python-docx')
a=parser.parse_args()
if a.rerun:
 for script in ['train.py','development_checks.py','evaluate.py','exploratory.py','audit.py']:
  subprocess.run([sys.executable,str(ROOT/'code'/script)],check=True)
records=json.loads((ROOT/'data/primary_observations.json').read_text());error=0.
for one in records:
 ep=episode(one['seed']);yl,yr,sigma=modified_observation(ep,one['case'])
 stored_l=np.array(one['yl_re'])+1j*np.array(one['yl_im']);stored_r=np.array(one['yr_re'])+1j*np.array(one['yr_im'])
 error=max(error,float(np.max(abs(yl-stored_l))),float(np.max(abs(yr-stored_r))))
assert error<1e-12,error
p=pd.read_csv(ROOT/'data/primary_results.csv')
assert len(p)==6400 and len(records)==1280
assert p.groupby(['case','method']).size().eq(256).all()
assert p[p['case']=='Nominal'].groupby('true_label').size().eq(320).all()
ex=pd.read_csv(ROOT/'data/exchange_results.csv')
assert len(ex)==1280 and ex.seed.nunique()==128
assert ex.groupby(['method','seed']).size().eq(2).all()
coordinate_differences={}
for col in ['log_b_estimate','log_p_estimate','logit_k_estimate']:
 wide=p[p['case']=='Nominal'].pivot(index='seed',columns='method',values=col)
 for m in ['RefCold','RefAnalytic']:
  value=float(np.max(abs(wide.RefLearned-wide[m])));coordinate_differences[col+'_'+m]=value;assert value<1e-3
subprocess.run([sys.executable,str(ROOT/'code/audit.py')],check=True)
report=dict(primary_observations_replayed=len(records),maximum_replay_difference=error,primary_rows=len(p),exchange_rows=len(ex),refined_coordinate_differences=coordinate_differences)
(ROOT/'data/replay_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if a.build_documents:
 for script in ['assemble_references.py','figures.py','build_documents.py']:
  subprocess.run([sys.executable,str(ROOT/'code'/script)],check=True)
