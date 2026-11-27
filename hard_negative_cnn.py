"""Train isolated 32px CNN rejection heads on CUDA, with group-held-out diagnostics."""
import hashlib
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn

from hard_negative_data import OUT


def network(classes):
    return nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Flatten(),nn.Dropout(.2),nn.Linear(768,classes))


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with np.load(OUT/'data.npz',allow_pickle=False) as d:
        x=torch.from_numpy(d['x']).cuda()[:,None].float()/255
        y=torch.as_tensor(d['y'],device='cuda',dtype=torch.long);groups=d['groups']
    hold=np.isin(groups,np.unique(groups)[::5]); assert not set(groups[hold])&set(groups[~hold])
    logs={}
    for n in (2,37):
        targets=(y==36).long() if n==2 else y
        def fit(use,count,monitor):
            torch.manual_seed(42); rng=np.random.default_rng(42); model=network(n).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            counts=torch.bincount(targets[use],minlength=n).clamp(min=1).float()
            loss=nn.CrossEntropyLoss(weight=counts.rsqrt())
            history=[]
            for _ in range(count):
                model.train()
                for ids in np.array_split(rng.permutation(use),int(np.ceil(len(use)/64))):
                    optimizer.zero_grad(set_to_none=True); loss(model(x[ids]),targets[ids]).backward();optimizer.step()
                if monitor:
                    model.eval()
                    with torch.inference_mode():history.append(float(loss(model(x[hold]),targets[hold])))
            return model,history
        start=perf_counter();torch.cuda.reset_peak_memory_stats()
        _,hist=fit(np.flatnonzero(~hold),16,True); epochs=int(np.argmin(hist))+1
        model,_=fit(np.arange(len(groups)),epochs,False)
        path=OUT/f'cnn_{n}.pt'
        with path.open('xb') as f:torch.save(model.state_dict(),f)
        logs[n]={'epochs':epochs,'holdout_loss':hist,'peak_reserved':torch.cuda.max_memory_reserved(),
                 'seconds':perf_counter()-start,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'cnn_training.json').write_text(json.dumps(logs,indent=2));print(logs)


if __name__=='__main__':main()