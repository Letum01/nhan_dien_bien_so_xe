"""Gated pilot. Default invocation checks readiness only, never fits."""
import argparse
import json
import shutil
from time import perf_counter
from pathlib import Path

from glyph_review import ROOT
from annotation_gate import check_gate
from yolo_character_dataset import ReadOnlyStore, sha, validate
from training_artifacts import artifact_root, resolve_artifact

PROTOCOL = resolve_artifact('reports/hard13_v2_protocol.json')
PRETRAINED = resolve_artifact('pretrained/yolov8n.pt')
CONFIG = ROOT/'configs/yolo_character_pilot.json'


def readiness(dataset, receipt=None):
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for key in ('code_sha256','checkpoints_sha256'):
        for path,digest in protocol[key].items():
            resolved = (ROOT/path) if key == 'code_sha256' else resolve_artifact(path, digest)
            if sha(resolved)!=digest:
                raise ValueError(f'BLOCKED: hard13 frozen hash changed: {path}')
    reviewed=ReadOnlyStore(resolve_artifact('data/ocr_glyphs.sqlite3')).all_latest()
    selected=check_gate(protocol,reviewed)
    if receipt is None:
        raise ValueError('BLOCKED: run frozen hard13 benchmark and supply its results JSON')
    receipt=Path(receipt)
    result=json.loads(receipt.read_text(encoding='utf-8'))
    keys={(r['image_path'],r['source_group'],c,v) for r in selected for c in ('gt','yolo') for v in protocol['variants']}
    records=[(r['image'],r['group'],r['crop'],r['variant']) for r in result['records']]
    if result['protocol']!=protocol or set(records)!=keys or len(records)!=len(keys):
        raise ValueError('BLOCKED: incomplete hard13 benchmark or different protocol')
    snapshot=json.loads((receipt.parent/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    by_id={r['image_id']:r for r in snapshot}
    if any(by_id.get(r['image_id'])!=r for r in selected):
        raise ValueError('BLOCKED: hard13 reviewed revisions differ from benchmark snapshot')
    return validate(dataset, source_root=artifact_root())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--hard13-result',type=Path)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--batch',type=int,choices=(1,2,4),default=4)
    args=parser.parse_args()
    try:
        result=readiness(args.dataset,args.hard13_result)
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,f'BLOCKED: {exc}\n')
    print(json.dumps(result,indent=2))
    if not args.train:
        return
    # Preflight can validate a historical root by rebasing it, but Ultralytics
    # must never receive an immutable backup YAML still pointing to the old root.
    yaml_root=json.loads((args.dataset/'data.yaml').read_text(encoding='utf-8')).get('path')
    if yaml_root != str(args.dataset.resolve()):
        parser.error('Export a new writable dataset/YAML with its actual root before fitting; do not modify the historical snapshot')
    base=ROOT/'experiments'
    dst=args.output.resolve() if args.output else None
    if not dst or dst.parent!=base or dst.exists():
        parser.error('--output must be a new direct child of experiments/')
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA required; no CPU fallback')
    if not PRETRAINED.is_file():
        parser.error('Trusted local pretrained/yolov8n.pt is required')
    model=YOLO(str(PRETRAINED))
    args.dataset=args.dataset.resolve()
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    config['batch']=args.batch
    if any(config[k]!=0 for k in ('fliplr','flipud')):
        parser.error('Flips forbidden for glyphs')
    receipt={'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'hard13_result_sha256':sha(args.hard13_result),
             'pretrained_sha256':sha(PRETRAINED),'config':config,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    # Do not create the run directory before Ultralytics resolves the new run name.
    provenance=dst.with_suffix('.inputs.json')
    with provenance.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    torch.cuda.reset_peak_memory_stats(0)
    start=perf_counter()
    one_batch={}
    def capture_batch(trainer):
        if not one_batch:
            one_batch.update(batch=int(trainer.batch_size),amp=bool(trainer.amp),device=str(trainer.device),
                             loader_batch_size=trainer.train_loader.batch_size)
            (dst/'first_batch.json').write_text(json.dumps(one_batch,indent=2),encoding='utf-8')
    model.add_callback('on_train_batch_end',capture_batch)
    model.train(data=str(args.dataset/'data.yaml'),project=str(base),name=dst.name,exist_ok=False,**config)
    shutil.copyfile(args.dataset/'manifest.json',dst/'dataset_manifest.json')
    metrics={'seconds':perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'first_batch':one_batch,
             'weights':{name:sha(dst/'weights'/name) for name in ('best.pt','last.pt')}}
    (dst/'training_receipt.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()