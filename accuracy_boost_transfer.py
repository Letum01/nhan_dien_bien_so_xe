"""Matched CUDA experiment: UCI backbone init vs scratch, no development tuning."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from final_ocr_dataset import ROOT, check_protocol, read, sha
from ocr_v3_experiment import edits as edit_distance, network
from progressive_train import metrics
from external_dataset_prepare import write
from glyph_review import CLASSES

CACHE=ROOT/'experiments/hybrid_features_01'
UCI=ROOT/'external_datasets/uci_optdigits/study_v1/cnn'
EPOCHS=12
BATCH=64
SEED=42


def fit(x, y, backbone):
    torch.manual_seed(SEED)
    model=network().cuda()
    if backbone is not None:
        model[:-1].load_state_dict(backbone)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=BATCH,
                      shuffle=True,generator=torch.Generator().manual_seed(SEED),pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loss_fn=nn.CrossEntropyLoss(); scaler=torch.amp.GradScaler('cuda'); history=[]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=perf_counter()
    model.train()
    for _ in range(EPOCHS):
        total=0.
        for batch,labels in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logits=model(batch.cuda(non_blocking=True)[:,None].float()/255)
                loss=loss_fn(logits,labels.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*len(labels)
        history.append(total/len(y))
    torch.cuda.synchronize()
    return model,dict(train_loss=history,seconds=perf_counter()-t,peak_allocated=torch.cuda.max_memory_allocated())


def predict(model,x):
    model.eval(); torch.cuda.synchronize(); t=perf_counter()
    with torch.inference_mode():
        ids=torch.cat([model(b.cuda()[:,None].float()/255).argmax(1).cpu() for b in torch.from_numpy(x).split(BATCH)])
    torch.cuda.synchronize()
    return ids.numpy(),perf_counter()-t


def main(out):
    check_protocol()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    meta=read(CACHE/'samples.json'); pixels=CACHE/'pixels.npy'; init=UCI/'best.pt'
    if sha(pixels)!=meta['pixels_sha256'] or sha(init)!=read(UCI/'development.json')['checkpoint_sha256']:
        raise ValueError('Existing pixel snapshot or UCI checkpoint changed')
    rows=meta['samples']; x=np.load(pixels,allow_pickle=False)
    counts=Counter(r['domain'] for r in rows)
    if counts!=dict(train=370,detector_train=344,synthetic=1800,validation=99,detector_dev=100) or x.shape!=(2713,32,32) or x.dtype!=np.uint8:
        raise ValueError('Unexpected cached sample contract')
    train=np.array([i for i,r in enumerate(rows) if r['domain'] in ('train','synthetic','detector_train')]); dev=np.array([i for i,r in enumerate(rows) if r['domain'] in ('validation','detector_dev')])
    for field in ('group','parent_sha256'):
        if {rows[i][field] for i in train if rows[i][field]} & {rows[i][field] for i in dev}:
            raise ValueError(f'Train/development overlap: {field}')
    y=np.array([CLASSES.index(r['label']) if r['label'] is not None else -1 for r in rows],dtype=np.int64)
    plates=meta['plates']; saved=read(ROOT/'reports/academic_development_replay.json'); old={r['id']:r for r in saved}
    if (len(plates)!=12 or set(old)!={p['id'] for p in plates} or
        sorted(i for p in plates for i in p['ids'])!=[i for i in dev if rows[i]['domain']=='detector_dev'] or
        any(p['reference']!=old[p['id']]['reference'] or p['baseline']!=old[p['id']]['ocr']['cnn']['prediction'] for p in plates)):
        raise ValueError('Development cohort does not match fixed replay')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(hypothesis='Transfer from licensed UCI handwritten digits may improve digit strokes',seed=SEED,batch=BATCH,
          epochs=EPOCHS,optimizer='AdamW defaults, LR .001',amp=True,classes=CLASSES,domains=dict(counts),
          train_policy='Existing cached samples; no new augmentation or external ALPR dataset; same indices in both arms',
          selection='Fixed last epoch, no tuning/selection on old development; promotion requires a new approved cohort',
          test_inference=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__,hashes={str(p.relative_to(ROOT)):sha(p) for p in
              (pixels,CACHE/'samples.json',init,ROOT/'reports/academic_development_replay.json',Path(__file__).resolve())}))
    backbone={k:v for k,v in torch.load(init,map_location='cpu',weights_only=True).items() if not k.startswith('14.')}
    results={};
    for key in ('scratch','uci_init'):
        model,info=fit(x[train],y[train],backbone if key=='uci_init' else None)
        path=out/f'{key}.pt'; torch.save(model.state_dict(),path)
        model=network().cuda(); model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        pred,seconds=predict(model,x[dev]); lookup=dict(zip(dev,pred)); table=[]
        for p in plates:
            text=''.join(CLASSES[lookup[i]] for i in p['ids']); ref=p['reference']; row=old[p['id']]
            table.append(dict(id=p['id'],layout=row['layout'],raw=text,refined=text,reference=ref,baseline=p['baseline'],
                              exact=text==ref,edits=edit_distance(ref,text),adjustment=None))
        groups={d:np.array([i for i in dev if rows[i]['domain']==d and y[i]>=0]) for d in ('validation','detector_dev')}
        results[key]={**info,'checkpoint_sha256':sha(path),'checkpoint_reload':True,'ocr_only_batch_seconds':seconds,
                      'exact':sum(p['exact'] for p in table),'cer':sum(p['edits'] for p in table)/sum(len(p['reference']) for p in table),
                      'fixed':sum(p['exact'] and p['baseline']!=p['reference'] for p in table),
                      'regressed':sum(not p['exact'] and p['baseline']==p['reference'] for p in table),
                      'glyph':{d:metrics(y[ids],np.array([lookup[i] for i in ids])) for d,ids in groups.items()},'plates':table}
        write(out/'results.json',results)
        print(key,{k:results[key][k] for k in ('exact','cer','fixed','regressed','seconds','peak_allocated')},flush=True)
        del model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    main(parser.parse_args().output)