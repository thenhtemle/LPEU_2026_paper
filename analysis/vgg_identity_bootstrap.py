import numpy as np, json
from sklearn.linear_model import LogisticRegression
d=np.load('signals.npz'); M=['Original','Retrain','FineTune','NegGrad','LPEU','LPEU-noDiv']
fk=d['forget_key']; uk=d['unseen_test_key']
fids=np.unique(fk); uids=np.unique(uk)
fidx={k:np.where(fk==k)[0] for k in fids}; uidx={k:np.where(uk==k)[0] for k in uids}
def fs(a,b):
    k=min(len(a),len(b)); a=a[np.linspace(0,len(a)-1,k).astype(int)]; b=b[np.linspace(0,len(b)-1,k).astype(int)]
    X=np.r_[a,b].reshape(-1,1); y=np.r_[np.ones(k),np.zeros(k)]
    return abs(0.5-LogisticRegression(max_iter=500).fit(X,y).score(X,y))
point={m:fs(d[m+'__forget_conf'],d[m+'__unseen_conf']) for m in M}
rng=np.random.default_rng(0); B=1000; S={m:[] for m in M}
for _ in range(B):
    fi=np.concatenate([fidx[k] for k in rng.choice(fids,len(fids))]); ui=np.concatenate([uidx[k] for k in rng.choice(uids,len(uids))])
    for m in M: S[m].append(fs(d[m+'__forget_conf'][fi],d[m+'__unseen_conf'][ui]))
S={m:np.array(v) for m,v in S.items()}
out={'n_forget_ids':len(fids),'n_unseen_ids':len(uids),'B':B,'point':point,
 'ci':{m:[float(np.percentile(S[m],2.5)),float(np.percentile(S[m],97.5))] for m in M},'delta_vs_Original':{},'delta_vs_Retrain':{}}
for m in M:
    for ref,key in [('Original','delta_vs_Original'),('Retrain','delta_vs_Retrain')]:
        if m==ref: continue
        x=S[m]-S[ref]; out[key][m]={'mean':float(x.mean()),'ci':[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))],'P_gt0':float((x>0).mean())}
json.dump(out,open('vgg_v2_identity_bootstrap.json','w'),indent=1); print(json.dumps(out,indent=1))
