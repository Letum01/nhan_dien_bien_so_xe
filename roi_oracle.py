"""Development-only diagnostic; never imported by production inference."""
import json
from time import perf_counter

import cv2

from acceptance_pilot import edit_counts
from alpr_recovery import classify
from dual_yolo_client import Worker
from dual_yolo_pipeline import run
from final_ocr_dataset import ROOT, check_protocol, read
from glyph_review import normalized_text
from segmentation_benchmark import match_boxes
from yolo_character_dataset import validate
from yolo_character_pipeline import MixClassifiers, normalize_glyph, order_boxes

OUTPUT = ROOT/'reports/pixel_roi_oracle.json'


def ordered_indices(boxes):
    return [r['index'] for r in order_boxes([{'bbox':b,'index':i} for i,b in enumerate(boxes)])]


def main():
    check_protocol()
    dataset=ROOT/'datasets/yolo_character_round1'; validate(dataset)
    reviewed=[e['review'] for e in read(dataset/'manifest.json')['rows'] if e['review']['split']=='val']
    expected={r['image']:r for r in read(ROOT/'reports/accuracy_recovery/benchmark.json')['records'] if r['arm']=='baseline'}
    if len(reviewed)!=12 or len(expected)!=12:raise ValueError('Expected fixed development cohort')
    models=MixClassifiers(); records=[]
    with Worker() as worker:
        for row in reviewed:
            name=row['image_path']; cached=expected[name]; image=cv2.imread(str(ROOT/name))
            if image is None:raise ValueError(f'Cannot decode {name}')
            target=normalized_text(row['plate_text'])
            x,y,_,_=row['plate_bbox']; chars=[g['label'] for g in row['characters']]
            gt=[[x+a,y+b,w,h] for a,b,w,h in [g['bbox'] for g in row['characters']]]
            raw=run(image,worker,models,'all'); plate=raw['plate']; tick=perf_counter()
            fx,fy,*_=plate['crop_box'] if plate else (0,0)
            det=[[fx+a,fy+b,w,h] for a,b,w,h in raw['glyph_boxes']]
            if det!=cached['pred_boxes'] or any(t!=cached['ocr'][k]['prediction'] for k,t in raw['text'].items()):
                raise ValueError(f'V1 replay differs for {name}')
            geometry=match_boxes(det,gt); matches=geometry['matches']; idx=ordered_indices(gt)
            substituted=[b.copy() for b in det]
            for m in matches:substituted[m['pred']]=gt[m['gt']]
            def predict(boxes):
                return classify(models,[normalize_glyph(image[b:b+h,a:a+w]) for a,b,w,h in boxes])[0]
            outputs={'A_detector':raw['text'],'B_gt_boxes':predict([gt[i] for i in idx]),
                     'C_matched_replacement':predict(substituted),'D_gt_classes_pipeline_order':{'cnn':''.join(chars[i] for i in idx)}}
            record={'image':name,'group':row['source_group'],'reference':target,'gt_classes':chars,'gt_order':idx,
                    'det_boxes':det,'gt_boxes':gt,'substituted_boxes':substituted,'geometry':geometry,'output':outputs,
                    'v1_seconds':raw['timings']['total'],'cold':raw['cold'],'v1_peak_reserved':raw['memory']['peak_reserved'],
                    'oracle_seconds':perf_counter()-tick}
            record['matched_correct']={k:{b:sum(t[m['pred']]==chars[m['gt']] for m in matches) for b,t in outputs[k].items()}
                                       for k in ('A_detector','C_matched_replacement')}
            record['gt_correct']={k:sum(t[j]==chars[i] for j,i in enumerate(idx)) for k,t in outputs['B_gt_boxes'].items()}
            records.append(record)
    summary={}
    for arm in records[0]['output']:
        summary[arm]={}
        for backend in records[0]['output'][arm]:
            pairs=[(r['reference'],r['output'][arm][backend],r['output']['A_detector'][backend]) for r in records]
            n=sum(len(a) for a,b,c in pairs)
            summary[arm][backend]={'exact':sum(a==b for a,b,c in pairs),'characters':n,
                 'cer':sum(sum(edit_counts(a,b).values()) for a,b,c in pairs)/n,
                 'fixed':sum(a==b and a!=c for a,b,c in pairs),'regressions':sum(a!=b and a==c for a,b,c in pairs)}
            if arm in ('A_detector','C_matched_replacement'):
                summary[arm][backend].update(glyph_correct=sum(r['matched_correct'][arm][backend] for r in records),
                                             glyph_total=sum(r['geometry']['tp'] for r in records))
            if arm=='B_gt_boxes':summary[arm][backend].update(glyph_correct=sum(r['gt_correct'][backend] for r in records),glyph_total=n)
    a=summary['A_detector']['cnn'];c=summary['C_matched_replacement']['cnn']
    go=c['exact']>a['exact'] and c['cer']<=a['cer']
    out={'scope':'Development oracle only; GT never used in deployed inference','replay_matches':len(records),'summary':summary,
         'roi_gate':go,'gate_rule':'matched replacement must strictly improve CNN exact without increasing CER','records':records}
    with OUTPUT.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({'summary':summary,'roi_gate':go},indent=2))


if __name__=='__main__':main()