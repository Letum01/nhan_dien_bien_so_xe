"""Two-phase cross-env YOLO crop → legacy OCR smoke, no accuracy claim."""
import argparse
import json
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'experiments/yolo_round2_splitB_run01/integration_smoke'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['detect','recognize']);args=parser.parse_args()
    if args.phase=='detect':
        from yolo_detector import detect_yolo
        from ultralytics import YOLO
        checkpoint=ROOT/'experiments/yolo_round2_splitB_run01/weights/best.pt'
        last=YOLO(str(checkpoint.with_name('last.pt')))
        assert last.names=={0:'license_plate'}
        comparison=json.loads((checkpoint.parent.parent/'validation_comparison/comparison.json').read_text(encoding='utf-8'))
        FOLDER.mkdir(exist_ok=False)
        rows=[]
        for index,image in enumerate([r['image'] for r in comparison['per_image'] if r['model']=='round2']):
            decoded=cv2.imread(str(ROOT/image))
            detections=detect_yolo(decoded,str(checkpoint),confidence=.4)
            for j,(crop,box,score) in enumerate(detections):
                path=FOLDER/f'{index:02d}_{j:02d}.png';cv2.imwrite(str(path),crop)
                rows.append({'image':image,'crop':str(path),'box':box,'score':score,'scope':'validation not in training; correctness not assumed'})
        (FOLDER/'crops.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print('last.pt loaded, adapter crop count',len(rows))
    else:
        from main import load_trained_models
        from segmentation import segment_characters
        from ocr_backends import recognize_crop
        models=load_trained_models(str(ROOT/'saved_models'))
        rows=json.loads((FOLDER/'crops.json').read_text(encoding='utf-8'))
        for row in rows:
            crop=cv2.imread(row['crop']);segments=segment_characters(crop)
            row['glyph_count']=len(segments)
            row['ocr']={name:recognize_crop(crop,name,models,segments).to_dict() for name in ('svm','rf','cnn','ensemble')}
        (FOLDER/'results.json').write_text(json.dumps({'scope':'technical crop/segmentation/classifier smoke only; no OCR accuracy',
                                                    'results':rows},indent=2),encoding='utf-8')
        print('OCR smoke',len(rows),'crops','backend errors',sum(x['status']=='error' for r in rows for x in r['ocr'].values()))


if __name__=='__main__':main()