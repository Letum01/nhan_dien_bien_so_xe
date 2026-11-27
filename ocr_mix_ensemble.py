"""Opt-in majority evaluation of three runs on identical validation ordering."""
import argparse
import json
from pathlib import Path

from ensemble_recognition import vote


def evaluate(paths,output):
    records={name:json.loads(Path(path).read_text(encoding='utf-8')) for name,path in paths.items()}
    hashes={r['dataset_manifest_sha256'] for r in records.values()}
    expected={tuple(r['expected']) for r in records.values()}
    if len(hashes)!=1 or len(expected)!=1:raise ValueError('Different validation manifest/order across runs')
    result=vote({name:''.join(row['predictions']) for name,row in records.items()})
    truth=next(iter(expected))
    report={'method':'majority','correct':sum(a==b for a,b in zip(truth,result.raw_text)),
            'samples':len(truth),'accuracy':sum(a==b for a,b in zip(truth,result.raw_text))/len(truth),
            'uncertain_positions':result.uncertain_positions,'raw':result.raw_text,
            'runs':{k:str(v) for k,v in paths.items()},'scope':'Same development validation, not independent test',
            'weighted_soft':'NOT_ENABLED: no separately frozen validation weight/calibration artifacts'}
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('svm','rf','cnn'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(evaluate({k:getattr(args,k) for k in ('svm','rf','cnn')},args.output))