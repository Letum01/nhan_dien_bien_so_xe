"""Train a single approved progressive level; use synthetic tensors for CUDA smoke only."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from external_task_curation import CLASSES, crop, digest, load_image, read, safe_path, write
from progressive_ocr import approved_rows, nested
from yolo_character_pipeline import normalize_glyph


def pixels(rows, ext):
    result=[]; parent=None; image=None
    for row in rows:
        if row['split'] not in ('train','development'):
            raise ValueError('Test pixels must not reach training')
        if row['parent']!=parent:
            parent=row['parent']; path=safe_path(ext/'OCR_Data',parent)
            if digest(path)!=row['sha256']: raise ValueError(f'Changed image: {parent}')
            image=load_image(path)
        result.append(normalize_glyph(crop(image,row['pixel_xywh'])[0]))
    return np.stack(result),np.array([r['target_id'] for r in rows],dtype=np.int64)


def unit_mean_weights(y):
    freq=Counter(y.tolist())
    weights=np.array([1/freq[k]**.5 for k in y])
    return weights/weights.mean()


def metrics(y, pred):
    cm=np.zeros((36,36),np.int64)
    np.add.at(cm,(y,pred),1)
    support=cm.sum(1); pp=cm.sum(0); hit=cm.diagonal()
    recall=np.divide(hit,support,out=np.zeros(36),where=support>0)
    f1=np.divide(2*hit,support+pp,out=np.zeros(36),where=support+pp>0)
    return dict(accuracy=float(hit.sum()/len(y)),macro_f1=float(f1[support>0].mean()),
                recall={c:float(recall[i]) if support[i] else None for i,c in enumerate(CLASSES)},
                support=dict(zip(CLASSES,support.tolist())),confusion_matrix=cm.tolist())


def cnn(train, dev, out, batch, epochs, contract, resume=False):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from ocr_v3_experiment import network

    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    torch.manual_seed(42)
    generator=torch.Generator().manual_seed(42)
    def loader(data, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(data[0]),torch.as_tensor(data[1],dtype=torch.long)),batch_size=batch,shuffle=shuffle,
                          generator=generator if shuffle else None,pin_memory=True,num_workers=0)
    tr,va=loader(train,True),loader(dev,False)
    net=network().cuda(); opt=torch.optim.AdamW(net.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3)
    scale=torch.amp.GradScaler('cuda'); criterion=nn.CrossEntropyLoss()
    history=[]; best=float('inf'); wait=0; first=0; elapsed=0.
    latest=out/'last.pt'; best_path=out/'best.pt'
    if resume:
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['contract']!=contract: raise ValueError('Resume contract mismatch')
        net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        scale.load_state_dict(state['amp']); generator.set_state(state['generator'])
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        first,best,wait,history,elapsed=(state[k] for k in ('epoch','best','wait','history','seconds'))
        best_weights=state['best_weights']
    def save(payload,path):
        tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
    def forward(x): return net(x.cuda(non_blocking=True)[:,None].float()/255)
    def eval_current():
        pred=[]; total=0.; net.eval()
        with torch.inference_mode():
            for x,y in va:
                logits=forward(x); total+=criterion(logits,y.cuda(non_blocking=True)).item()*len(y)
                pred.extend(logits.argmax(1).cpu().tolist())
        return total/len(dev[1]),np.asarray(pred)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=perf_counter()
    for epoch in range(first,epochs) if wait<10 else ():
        net.train(); total=0.
        for x,y in tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss=criterion(forward(x),y.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite CUDA loss')
            scale.scale(loss).backward(); scale.step(opt); scale.update(); total+=loss.item()*len(y)
        val,_=eval_current(); sched.step(val)
        history.append(dict(epoch=epoch+1,train_loss=total/len(train[1]),validation_loss=val,lr=opt.param_groups[0]['lr']))
        if val<best:
            best=val; wait=0; best_weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else: wait+=1
        save(dict(contract=contract,model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),amp=scale.state_dict(),
                  generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                  epoch=epoch+1,best=best,best_weights=best_weights,wait=wait,history=history,seconds=elapsed+perf_counter()-start),latest)
        if wait>=10: break
    net.load_state_dict(best_weights); save(best_weights,best_path); _,pred=eval_current()
    torch.cuda.synchronize()
    return pred,dict(epochs=len(history),history=history,training_seconds=elapsed+perf_counter()-start,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),amp=True)


def classical(train, dev, groups, backend, feature, output):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from dataset import extract_hog_features
    from hybrid_features import extract

    combined=np.concatenate([train[0],dev[0]]); parts=[]; extractor=None
    if feature in ('hog','fusion'):
        hog=np.stack([extract_hog_features(im) for im in combined]); assert hog.shape[1]==324
        parts.append(hog)
    if feature in ('embedding','fusion'):
        arrays,extractor=extract('v1',combined); parts.append(arrays['embedding'])
    x=np.concatenate(parts,axis=1); y=train[1]; xt,xd=x[:len(y)],x[len(y):]
    if not np.isfinite(x).all(): raise ValueError('Nonfinite features')
    def model():
        if backend=='rf': return RandomForestClassifier(n_estimators=200,max_depth=24,n_jobs=4,random_state=42)
        return make_pipeline(StandardScaler(), SVC(C=3,kernel=backend.removeprefix('svm-')))
    def fit(m,a,b):
        kwargs={'sample_weight' if backend=='rf' else 'svc__sample_weight':unit_mean_weights(b)}
        return m.fit(a,b,**kwargs)
    stamp=perf_counter(); folds=[]
    n=min(3,len(set(groups)))
    if n>=2:
        for a,b in GroupKFold(n).split(xt,y,groups):
            if len(set(y[a]))<2: folds.append(None); continue
            folds.append(metrics(y[b],fit(model(),xt[a],y[a]).predict(xt[b]))['accuracy'])
    fitted=fit(model(),xt,y); seconds=perf_counter()-stamp
    joblib.dump(fitted,output/'classifier.joblib',compress=3)
    return fitted.predict(xd),dict(training_seconds=seconds,group_fold_accuracy=folds,feature_dim=x.shape[1],
                                   extractor=extractor,sample_weight_mean=float(unit_mean_weights(y).mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('curation',type=Path); ap.add_argument('--approval',type=Path)
    ap.add_argument('--ext',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--level',type=int,choices=(1000,5000,10000,15000),required=True)
    ap.add_argument('--backend',choices=('cnn','svm-rbf','svm-linear','rf'),required=True)
    ap.add_argument('--features',choices=('hog','embedding','fusion'),default='hog')
    ap.add_argument('--batch',type=int,choices=(32,64,128),default=32);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    try:
        rows,frozen,protocol=approved_rows(a.curation,a.approval)
        selected=nested(rows,(a.level,)).get(a.level)
        if not selected: raise ValueError('Insufficient approved glyphs for this level')
        dev=[r for r in rows if r['split']=='development']
        contract=dict(frozen=frozen,protocol=digest(protocol),level=a.level,batch=a.batch,backend=a.backend,features=a.features,
                      curation=digest(a.curation/'glyphs.json'),code=digest(Path(__file__)),sampler=digest(Path(__file__).with_name('progressive_ocr.py')))
        if a.resume and a.backend!='cnn': raise ValueError('Resume supported for CNN only')
        if a.resume and read(a.output/'contract.json')!=contract: raise ValueError('Resume contract mismatch')
        tr,va=pixels(selected,a.ext),pixels(dev,a.ext)
        if len(set(tr[1]))<2: raise ValueError('Need at least two training classes')
        if not a.resume:
            a.output.mkdir(parents=True,exist_ok=False); write(a.output/'contract.json',contract)
        if a.backend=='cnn': pred,extra=cnn(tr,va,a.output,a.batch,60,contract,a.resume)
        else:
            # Same connected units as scaling, so duplicate components cannot straddle folds.
            from progressive_ocr import sampling_units
            groups=[None]*len(selected); lookup={(r['parent'],r['glyph_index']):i for i,r in enumerate(selected)}
            for key,unit in sampling_units(selected).items():
                for r in unit: groups[lookup[r['parent'],r['glyph_index']]]=key
            pred,extra=classical(tr,va,groups,a.backend,a.features,a.output)
        write(a.output/'development_metrics.json',{**metrics(va[1],pred),**extra,'glyph_count':len(selected),
              'backend':a.backend,'features':'cnn32' if a.backend=='cnn' else a.features,
              'benchmark_sha256':frozen,'test_inference':False,'contract':contract,
              'checkpoint_sha256':digest(a.output/('best.pt' if a.backend=='cnn' else 'classifier.joblib'))})
    except (OSError,ValueError,KeyError,RuntimeError) as e:
        ap.exit(2,f'{e}\n')


if __name__=='__main__': main()