from model import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib,json,time,warnings
def dataset(seed,n):
    rng=np.random.default_rng(seed);x=np.c_[rng.uniform(np.log(3),np.log(30),n),rng.uniform(np.log(3),np.log(45),n),logit(rng.uniform(.08,.92,n))];gain=rng.uniform(.6,1.4,(n,2));sigma=np.exp(rng.uniform(np.log(.004),np.log(.06),(n,2)));yl=[];yr=[]
    for j in range(n):
        l,r=observation(x[j],gain[j],seed*100000+j,sigma[j]);yl.append(l);yr.append(r)
    yl=np.array(yl);yr=np.array(yr);return dict(X=features(yl,yr,sigma),y=x,yl=yl,yr=yr,sigma=sigma,gain=gain)
def predict(bundle,yl,yr,sigmas):
    feat=features(yl,yr,sigmas);out=bundle['ys'].inverse_transform(bundle['model'].predict(bundle['xs'].transform(np.atleast_2d(feat))))
    return np.clip(out,LOW+1e-6,HIGH-1e-6)
def main():
    st=time.time();tr=dataset(4100,24000);va=dataset(4200,4000);np.savez_compressed(ROOT/'data/training.npz',**tr);np.savez_compressed(ROOT/'data/validation.npz',**va);xs=StandardScaler().fit(tr['X']);ys=StandardScaler().fit(tr['y']);m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=.001,batch_size=256,learning_rate_init=.001,max_iter=200,early_stopping=True,validation_fraction=.1,n_iter_no_change=12,random_state=41)
    m.fit(xs.transform(tr['X']),ys.transform(tr['y']));bundle=dict(model=m,xs=xs,ys=ys);joblib.dump(bundle,ROOT/'data/initializer.joblib');pred=predict(bundle,va['yl'],va['yr'],va['sigma']);rep=dict(seed=41,training=24000,validation=4000,iterations=m.n_iter_,parameters=sum(c.size for c in m.coefs_)+sum(b.size for b in m.intercepts_),validation_transformed_rmse=np.sqrt(np.mean((pred-va['y'])**2,0)).tolist(),validation_bandwidth_relative_median=np.median(abs(np.exp(pred[:,:2]-va['y'][:,:2])-1),0).tolist(),elapsed_seconds=time.time()-st,loss_curve=m.loss_curve_);(ROOT/'data/training_report.json').write_text(json.dumps(rep,indent=2));print(json.dumps({k:v for k,v in rep.items() if k!='loss_curve'},indent=2))
if __name__=='__main__':main()
