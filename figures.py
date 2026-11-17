"""Exact vector diagrams and scientific plots; every plotted value is computed."""
from model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Polygon,FancyBboxPatch,FancyArrowPatch,Ellipse
import pandas as pd,json
from pathlib import Path
from scipy.special import logit
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif'],'font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9.5,'xtick.labelsize':10,'ytick.labelsize':10,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.grid':False})
COL=plt.cm.viridis(np.linspace(.12,.85,5));MAN=[]
METHODS=['LinkMulti','RefCold','RefAnalytic','RefLearned','RefNetworkOnly'];LABELS=['Link','Ref\ncold','Ref\nanalytic','Ref\nlearned','Network\nonly']
short={'LinkMulti':'Link multistart','RefCold':'Reference cold','RefAnalytic':'Reference analytic','RefLearned':'Reference learned','RefNetworkOnly':'Network only'}
def save(fig,n,desc,diagram=False):
 fig.savefig(OUT/f'Fig{n}.png',dpi=1200 if diagram else 600,bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/f'Fig{n}.eps',bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/f'Fig{n}.svg',bbox_inches='tight',facecolor='white')
 MAN.append(dict(number=n,description=desc,raster_dpi=1200 if diagram else 600,vector=['eps','svg'],font='Liberation Serif (Times New Roman metric-compatible fallback); STIX math'))
 plt.close(fig)
def setup(ax,xlabel=None,ylabel=None):
 if xlabel:ax.set_xlabel(xlabel)
 if ylabel:ax.set_ylabel(ylabel)
 ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
def box(ax,x,y,w,h,text,fill='#edf2f4',fs=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.04',fc=fill,ec='#334155',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(ax,a,b,**kw):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.1,color='#334155',**kw))

# Fig 1: pictorial hardware paths and data interfaces, deliberately schematic.
fig,ax=plt.subplots(figsize=(7.0,3.7));ax.set(xlim=(0,10),ylim=(0,5.35));ax.axis('off')
ax.plot([.2,5.1,5.1],[4.85,4.85,.8],color='.5',lw=1.5);ax.plot([.2,5.1],[.8,.8],color='.5',lw=1.5)
ax.add_patch(Rectangle((.7,4.55),1.0,.3,fc='#566574',ec='black'));ax.add_patch(Ellipse((1.2,4.54),.7,.16,fc='#ffd166',ec='#6b5c2a'));ax.text(1.2,5.12,'LED and driver',ha='center',fontsize=10)
ax.add_patch(Polygon([[1.15,4.4],[2.9,1.4],[3.8,1.4]],fc='#f5edcb',ec='none'))
ax.add_patch(Rectangle((3.0,1.04),.65,.3,fc='#326f82',ec='black'));ax.add_patch(Rectangle((3.1,1.34),.45,.12,fc='#264653'));ax.text(3.32,.38,'Link photodiode',ha='center',fontsize=10)
arrow(ax,(1.35,4.35),(3.3,1.48),color=None) if False else None
arrow(ax,(1.35,4.3),(3.22,1.5));ax.text(2.0,2.75,'LOS',rotation=-54,fontsize=10)
ax.add_patch(Rectangle((4.86,2.9),.24,.7,fc='#e3e6e9',ec='#586474'));arrow(ax,(1.6,4.35),(4.8,3.2));arrow(ax,(4.82,3.16),(3.58,1.5));ax.text(4.08,3.92,'Diffuse paths',ha='center',fontsize=10)
ax.add_patch(Rectangle((.27,2.45),.56,.25,fc='#326f82',ec='black'));ax.add_patch(Rectangle((.35,2.7),.4,.1,fc='#264653'));arrow(ax,(.96,4.33),(.56,2.86));ax.text(.17,1.92,'Shielded\nreference PD',ha='left',fontsize=10)
box(ax,5.8,3.78,3.65,.95,'Raw complex spectra\n12 calibrated probes per receiver')
box(ax,5.8,2.12,3.65,1.03,'Physical twin and MLP initializer\nFour constrained hypothesis fits')
box(ax,5.8,.38,3.65,1.04,'Attribution or rejection\nParameters and local intervals')
arrow(ax,(5.15,3.0),(5.75,4.08));arrow(ax,(5.15,1.2),(5.75,4.0));arrow(ax,(7.62,3.75),(7.62,3.2));arrow(ax,(7.62,2.09),(7.62,1.48))
save(fig,1,'Schematic reference-assisted VLC measurement and inference',True)

# Fig 2: exactly indistinguishable link and distinguishable reference spectra.
f=np.linspace(0,38,501);a=np.array([np.log(7.8),np.log(12),logit(.4)]);b=swap(a);la,ra=transfer(a,f);lb,rb=transfer(b,f)
pd.DataFrame(dict(frequency_MHz=f,link_A_dB=20*np.log10(abs(la)),link_B_dB=20*np.log10(abs(lb)),reference_A_dB=20*np.log10(abs(ra)),reference_B_dB=20*np.log10(abs(rb)))).to_csv(OUT/'Fig2_data.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
for ax,h1,h2,ttl in [(axs[0],la,lb,'(a) Link spectra coincide'),(axs[1],ra,rb,'(b) Reference spectra separate')]:
 ax.plot(f,20*np.log10(abs(h1)),color=COL[0],lw=2,label='Emitter change');ax.plot(f,20*np.log10(abs(h2)),color=COL[3],lw=1.5,ls='--',label='Channel change');ax.set_title(ttl);setup(ax,'Probe frequency (MHz)','Magnitude (dB)');ax.legend(loc='lower left',frameon=False)
save(fig,2,'Exact pole-exchange counterexample')

# Fig 3: executable order and the rejection branch.
fig,ax=plt.subplots(figsize=(6.9,4.15));ax.set(xlim=(0,10),ylim=(0,6.3));ax.axis('off')
box(ax,2.6,5.42,4.8,.62,'Commissioned baseline\nand raw link/reference spectra')
box(ax,2.6,4.31,4.8,.66,'Normalize features\nInfer bounded MLP start')
for x,t in [(0.35,'H0  No change'),(2.85,'HE  Emitter'),(5.35,'HC  Channel'),(7.85,'HM  Mixed')]:
 box(ax,x,2.95,1.8,.7,t,fs=10);arrow(ax,(5,4.28),(x+.9,3.7))
box(ax,2.55,1.86,4.9,.63,'Fit raw-spectrum likelihood\nOrder four BIC values')
for x in [1.25,3.75,6.25,8.75]:arrow(ax,(x,2.92),(5,2.52))
diamond=Polygon([[5,1.7],[6.08,1.22],[5,.74],[3.92,1.22]],fc='#e8efe1',ec='#334155');ax.add_patch(diamond);ax.text(5,1.22,'Residual and\nmargin pass',ha='center',va='center',fontsize=9.5);arrow(ax,(5,1.84),(5,1.72))
box(ax,.45,.38,2.65,.7,'Reject attribution\nInspect calibration',fs=10);box(ax,6.88,.38,2.65,.7,'Report hypothesis\nand local intervals',fs=10)
arrow(ax,(3.9,1.22),(3.14,.75));arrow(ax,(6.1,1.22),(6.84,.75));ax.text(3.35,1.24,'No',fontsize=9.5);ax.text(6.38,1.24,'Yes',fontsize=9.5);arrow(ax,(5,5.4),(5,5.0))
save(fig,3,'Flowchart of reference learned estimator and failure checks',True)

P=pd.read_csv(ROOT/'data/primary_results.csv');S=pd.read_csv(ROOT/'data/primary_summary.csv');CI=pd.read_csv(ROOT/'data/bootstrap_intervals.csv');N=S[S['case']=='Nominal'].set_index('method').loc[METHODS]
fig,axs=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
for ax,metric,title in [(axs[0],'correct','(a) Hard attribution accuracy'),(axs[1],'accepted_correct','(b) Correct and accepted fraction')]:
 c=CI[(CI['case']=='Nominal')&(CI.metric==metric)].set_index('method').loc[METHODS];ax.bar(np.arange(5),c['mean'],color=COL);ax.errorbar(np.arange(5),c['mean'],yerr=np.array([c['mean']-c.low,c.high-c['mean']]),fmt='none',color='black',capsize=2,lw=.9);ax.set_xticks(np.arange(5),LABELS);ax.set_ylim(0,1.08);ax.set_title(title);setup(ax,ylabel='Fraction of 256 episodes')
save(fig,4,'Nominal hard and selective outcomes with pointwise bootstrap intervals')

EX=pd.read_csv(ROOT/'data/exchange_results.csv');EXCI=pd.read_csv(ROOT/'data/exchange_cluster_intervals.csv')
fig=plt.figure(figsize=(7,3.3),layout='constrained');gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.25])
for j,m in enumerate(['LinkMulti','RefLearned']):
 ax=fig.add_subplot(gs[0,j]);g=EX[EX.method==m];mat=np.array([[(g[g.true_label==t].predicted_label==q).mean() for q in [0,1,2,3]] for t in [1,2]])
 ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks([0,1,2,3],['H0','HE','HC','HM']);ax.set_yticks([0,1],['Emitter','Channel']);ax.set_xlabel('Predicted hypothesis');ax.set_title(('(a) Link multistart' if j==0 else '(b) Reference learned'))
 for y in range(2):
  for x in range(4):ax.text(x,y,f'{mat[y,x]:.2f}',ha='center',va='center',color='white' if mat[y,x]<.5 else 'black',fontsize=9)
ax=fig.add_subplot(gs[0,2]);sel=EXCI[EXCI.metric=='correct'].set_index('method').loc[METHODS];y=np.arange(5);ax.errorbar(sel['mean'],y,xerr=[sel['mean']-sel.low,sel.high-sel['mean']],fmt='o',color=COL[1],capsize=2);ax.set_yticks(y,['Link','Cold','Analytic','Learned','Network']);ax.invert_yaxis();ax.set_xlim(.3,1.04);ax.axvline(.5,color='.5',ls=':',lw=1);ax.set_title('(c) Pair cluster intervals');ax.set_xlabel('Hard attribution accuracy')
save(fig,5,'Complete-pair attribution audit and clustered uncertainty')

fig,axs=plt.subplots(1,2,figsize=(7,3),layout='constrained');x=np.arange(5)
for j,(n,tx) in enumerate([('log_b','log b'),('log_p','log p'),('logit_k','logit κ')]):
 axs[0].bar(x+(j-1)*.24,N[n+'_rmse'],.23,color=COL[j+1],label=tx)
 axs[1].plot(np.arange(4),N[n+'_coverage'].iloc[:4],marker=['o','s','^'][j],color=COL[j+1],label=tx)
axs[0].set_yscale('log');axs[0].set_xticks(x,LABELS);axs[0].set_title('(a) Full-model parameter errors');setup(axs[0],ylabel='RMSE in transformed coordinates');axs[0].legend(frameon=False)
axs[1].set_xticks(np.arange(4),LABELS[:4]);axs[1].axhline(.95,color='.5',ls=':',lw=1);axs[1].set_ylim(.5,1.02);axs[1].set_title('(b) Local interval coverage');setup(axs[1],ylabel='Empirical 95% interval coverage');axs[1].legend(frameon=False,loc='lower right')
save(fig,6,'Parameter estimation and local intervals; link intervals conditional on rank')

fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
for j,(case,label) in enumerate([('RefPoleError','Wrong reference pole'),('DiffuseDelay','Unmodelled diffuse delay')]):
 a0=S[S['case']==case].set_index('method').loc[METHODS];axs[0].bar(x+(j-.5)*.35,a0.coverage,.34,color=COL[1+j*2],label=label)
axs[0].set_xticks(x,LABELS);axs[0].set_ylim(0,1.02);axs[0].set_title('(a) Explanations passing both gates');setup(axs[0],ylabel='Accepted explanation fraction');axs[0].legend(frameon=False,fontsize=9,loc='upper right')
for j,(case,label) in enumerate([('Nominal','Nominal'),('RefPoleError','Wrong reference'),('DiffuseDelay','Diffuse delay')]):
 g=S[(S['case']==case)&(S.method=='RefLearned')].iloc[0];axs[1].plot([0,1,2],[g[n+'_coverage'] for n in ['log_b','log_p','logit_k']],marker=['o','s','^'][j],color=COL[j*2],label=label)
axs[1].set_xticks([0,1,2],['log b','log p','logit κ']);axs[1].set_ylim(-.03,1.03);axs[1].axhline(.95,color='.5',ls=':',lw=1);axs[1].set_title('(b) Reference learned local intervals');setup(axs[1],ylabel='Coverage of physical parameter');handles,_=axs[1].get_legend_handles_labels();axs[1].legend(handles,['Nominal','Ref. pole','Delay'],frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,columnspacing=.7,handlelength=1.3)
save(fig,7,'Out-of-model acceptance and interval failures; no causal accuracy metric')

T=pd.read_csv(ROOT/'data/exploratory_warmed_timing.csv').groupby(['method','seed'])[['milliseconds','calls']].median();med=T.groupby('method').median().loc[METHODS];q1=T.groupby('method').quantile(.25).loc[METHODS];q3=T.groupby('method').quantile(.75).loc[METHODS]
fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained');axs[0].bar(x,N.calls,color=COL);axs[0].set_title('(a) Primary model evaluations');setup(axs[0],ylabel='Mean evaluations per episode');axs[1].bar(x,med.milliseconds,color=COL);axs[1].errorbar(x,med.milliseconds,yerr=[med.milliseconds-q1.milliseconds,q3.milliseconds-med.milliseconds],fmt='none',color='black',capsize=3);axs[1].set_title('(b) Independent warmed timing');setup(axs[1],ylabel='Median episode time (ms)')
for ax in axs:ax.set_xticks(x,LABELS)
save(fig,8,'Primary functional cost and independent 32-episode host timing with IQR')

R=pd.read_csv(ROOT/'data/exploratory_reference_noise.csv');RR=R.groupby(['reference_sigma','method']).agg(accuracy=('correct','mean'),p_rmse=('p_error',lambda x:np.sqrt(np.mean(x*x))),b_rmse=('b_error',lambda x:np.sqrt(np.mean(x*x)))).reset_index();RR.to_csv(OUT/'Fig9_data.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
for m,col,marker,lab in [('RefLearned',COL[3],'o','Refined estimators'),('RefNetworkOnly',COL[0],'s','Network only')]:
 g=RR[RR.method==m];axs[0].plot(g.reference_sigma,g.accuracy,color=col,marker=marker,label=lab);axs[1].plot(g.reference_sigma,g.p_rmse,color=col,marker=marker,label=lab)
for ax in axs:ax.set_xscale('log');ax.set_xticks([.004,.012,.04,.12],['0.004','0.012','0.04','0.12']);setup(ax,'Reference noise σR');ax.legend(frameon=False,loc='best')
axs[0].set_ylim(.5,1.02);axs[0].set_ylabel('Hard attribution accuracy');axs[0].set_title('(a) Exploratory noise sensitivity');axs[1].set_ylabel('RMSE of log p');axs[1].set_title('(b) Channel parameter error')
save(fig,9,'Exploratory reference-noise sweep, 64 fresh episodes per point')

W=pd.read_csv(ROOT/'data/exploratory_weak_diffuse.csv');WW=W.groupby('kappa').agg(rmse=('p_error',lambda x:np.sqrt(np.mean(x*x))),bounds=('at_bound','mean')).reset_index();WW.to_csv(OUT/'Fig10a_data.csv',index=False);FI=pd.read_csv(ROOT/'data/exploratory_fisher_span.csv')
fig,axs=plt.subplots(1,2,figsize=(7,2.85),layout='constrained');axs[0].plot(WW.kappa.iloc[1:],WW.rmse.iloc[1:],'-o',color=COL[2],label='Within fit bounds');axs[0].plot(WW.kappa.iloc[:2],WW.rmse.iloc[:2],':',color='.4');axs[0].plot(.02,WW.rmse.iloc[0],'o',mfc='white',mec=COL[0],ms=7,label='κ = 0.02 outside bounds');axs[0].set_xscale('log');axs[0].set_yscale('log');axs[0].set_title('(a) Weak diffuse component');setup(axs[0],'True diffuse fraction κ','RMSE of log p');axs[0].legend(frameon=False,fontsize=9)
for ref,col,lab in [(False,COL[0],'Link only'),(True,COL[3],'Link and reference')]:
 g=FI[FI.reference==ref];axs[1].plot(g.maximum_frequency_MHz,g.log_p_se,'-o',color=col,label=lab)
axs[1].set_yscale('log');axs[1].set_title('(b) Local Fisher calculation');setup(axs[1],'Maximum probe frequency (MHz)','Local standard error of log p');axs[1].legend(frameon=False)
save(fig,10,'Exploratory weak-diffuse sweep and deterministic probe-span calculation')
(OUT/'figure_manifest.json').write_text(json.dumps(MAN,indent=2));print('Exported',len(MAN),'figures, each with raster and two vector formats')
