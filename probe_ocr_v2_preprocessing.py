"""Exploratory development ablation; unchanged boxes, no deployment selection."""
import json
import cv2
import numpy as np

from acceptance_pilot import edit_counts
from final_ocr_dataset import ROOT, read, check_protocol
from glyph_review import CLASSES
from yolo_character_pipeline import MixClassifiers, normalize_glyph


def main():
    check_protocol()
    out=ROOT/'experiments/ocr_v2_200/preprocessing_probe.json'
    if out.exists():raise FileExistsError(out)
    rows=read(ROOT/'experiments/ocr_v2_200/plate_development.json')['records']
    boxes={r['image']:r['pred_boxes'] for r in read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['records']
           if r['scope']=='yolo' and r['method']=='character_yolo'}
    model=MixClassifiers().cnn
    curves={'original':lambda g:g,'clahe':lambda g:cv2.createCLAHE(2,(4,4)).apply(g),
            'gamma_0.8':lambda g:np.rint(255*(g/255)**.8).astype(np.uint8),
            'median_3':lambda g:cv2.medianBlur(g,3)}
    result={}
    for name,fn in curves.items():
        samples=[]
        for row in rows:
            im=cv2.imread(str(ROOT/row['image'])); chars=[]
            for x,y,w,h in boxes[row['image']]:
                roi=im[y:y+h,x:x+w]; gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                chars.append(normalize_glyph(cv2.cvtColor(fn(gray),cv2.COLOR_GRAY2BGR)))
            output=''.join(CLASSES[i] for i in np.asarray(model(np.asarray(chars)[...,None],training=False)).argmax(1)) if chars else ''
            samples.append({'image':row['image'],'reference':row['reference'],'text':output,'edits':sum(edit_counts(row['reference'],output).values())})
        result[name]={'exact':sum(r['text']==r['reference'] for r in samples),
                      'cer':sum(r['edits'] for r in samples)/sum(len(r['reference']) for r in samples),'rows':samples}
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print({k:(r['exact'],r['cer']) for k,r in result.items()})


if __name__=='__main__':main()