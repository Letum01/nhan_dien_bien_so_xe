"""Native Ultralytics AP on GT-crop development validation only (.venv-yolo)."""
import argparse
import json
from pathlib import Path

from yolo_character_dataset import sha, validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','weights','output'): parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not args.weights.is_file() or args.output.exists():
        parser.error('Local trained character weights and a new output directory are required')
    validate(args.dataset)
    from ultralytics import YOLO
    model=YOLO(str(args.weights.resolve()))
    if model.task!='detect' or model.names!={0:'character'}:
        parser.error('Expected one-class character detection checkpoint')
    out=args.output.resolve()
    metrics=model.val(data=str((args.dataset/'data.yaml').resolve()),split='val',imgsz=640,batch=4,
                      conf=.001,iou=.7,max_det=40,device=0,project=str(out.parent),name=out.name,exist_ok=False)
    (out/'ap.json').write_text(json.dumps({'scope':'GT-crop development val only','map50':float(metrics.box.map50),
        'map50_95':float(metrics.box.map),'manifest_sha256':sha(args.dataset/'manifest.json'),
        'weights_sha256':sha(args.weights)},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()