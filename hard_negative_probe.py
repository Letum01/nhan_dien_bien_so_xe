"""Export CUDA predictions for an explicit batch of normalized glyphs."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from hard_negative_cnn import network
from hard_negative_data import OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    x=np.load(a.inputs,allow_pickle=False)
    if x.ndim!=3 or x.shape[1:]!=(32,32) or x.dtype!=np.uint8:raise ValueError('Invalid glyph input')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    info=json.loads((OUT/'cnn_training.json').read_text()); result={}
    for n in (2,37):
        path=OUT/f'cnn_{n}.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info[str(n)]['sha256']:raise ValueError('Checkpoint changed')
        m=network(n).cuda();m.load_state_dict(torch.load(path,weights_only=True));m.eval()
        t=torch.from_numpy(x).cuda()[:,None].float()/255
        times=[]
        with torch.inference_mode():
            m(t[:8]);torch.cuda.synchronize()
            for _ in range(5):
                begin=perf_counter();z=m(t);torch.cuda.synchronize();times.append(perf_counter()-begin)
        result[str(n)]={'predictions':z.argmax(1).cpu().tolist(),'batch_median_s':float(np.median(times)),
                        'peak_reserved':torch.cuda.max_memory_reserved()}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()