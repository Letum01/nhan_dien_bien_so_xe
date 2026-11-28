"""Frozen CNN probes. Torch execution is isolated in the CUDA venv."""
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from glyph_review import ROOT
from yolo_character_dataset import sha
from yolo_character_pipeline import WEIGHTS, PYTHON

V2 = ROOT/'experiments/ocr_v2_200'
V3 = ROOT/'experiments/ocr_v3_cuda_retry'
MODELS = {'v1': WEIGHTS['cnn']}


def extract(name, pixels):
    tick=perf_counter()
    path=MODELS[name]
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[1:]!=(32,32):
        raise ValueError('Expected N×32×32 uint8')
    output={k:[] for k in ('pool','embedding','prediction')}
    import tensorflow as tf
    net=tf.keras.models.load_model(path,compile=False)
    inner=net.layers[-1]
    fmap=[l for l in inner.layers if isinstance(l,tf.keras.layers.MaxPooling2D)][-1]
    dense=[l for l in inner.layers if isinstance(l,tf.keras.layers.Dense)][-2]
    scales=[l for l in inner.layers if isinstance(l,tf.keras.layers.Rescaling)]
    if len(scales)!=1 or not np.isclose(scales[0].scale,1/255) or scales[0].offset!=0:
        raise ValueError('Unexpected normalization')
    probe=tf.keras.Model(inner.inputs,[fmap.output,dense.output,inner.outputs[0]])
    def batch(x):
        f,e,p=probe(x[...,None],training=False)
        f=np.asarray(f)
        return f.mean((1,2)),np.asarray(e),np.asarray(p).argmax(1),f
    devices=tf.config.list_physical_devices('GPU'); mode='TensorFlow GPU visible' if devices else 'TensorFlow CPU'
    layers=[fmap.name,dense.name]; framework=tf.__version__
    load=perf_counter()-tick
    warm=batch(pixels[:min(8,len(pixels))]); times=[]; samples=[]
    if not np.array_equal(warm[2],np.asarray(net(pixels[:8,...,None],training=False)).argmax(1)):
        raise ValueError('Probe disagrees with full checkpoint')
    for i in range(0,len(pixels),128):
        t=perf_counter(); p,e,y,_=batch(pixels[i:i+128]);times.append(perf_counter()-t)
        for key,value in zip(output,(p,e,y)):output[key].append(value)
    latency=[]
    for _ in range(10):
        t=perf_counter();batch(pixels[:8]);latency.append(perf_counter()-t)
    meta={'checkpoint':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'framework':framework,
          'architecture':[(l.name,type(l).__name__,str(l.output.shape)) for l in inner.layers],
          'layers':dict(zip(('pool_source','embedding'),layers)),'device':mode,'load_seconds_includes_framework_import':load,
          'glyphs_per_second_compute':len(pixels)/sum(times),'batch8_median_ms':1000*float(np.median(latency)),
          'gpu_samples_util_pct_used_mib_temp_c':samples,
          'peak_reserved_bytes':None,
          'peak_allocated_bytes':None,
          'preprocessing':'32x32 uint8 white-on-black; one /255; augmentation disabled; mean pooling is spatial'}
    return {**{k:np.concatenate(v) for k,v in output.items()},'maps':warm[3]},meta


if __name__=='__main__':
    name,folder=sys.argv[1:]; folder=Path(folder)
    features,meta=extract(name,np.load(folder/'pixels.npy',allow_pickle=False))
    np.savez_compressed(folder/f'{name}_features.npz',**features)
    (folder/f'{name}_extract.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
