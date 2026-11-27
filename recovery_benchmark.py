"""Development-only benchmark and opt-in case traces; never fits models."""
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from alpr_recovery import VARIANTS, analyze, worker, pixels
from benchmark_yolo_character import score
from dataset_audit import fingerprint
from dual_yolo_client import Worker
from dual_yolo_pipeline import run
from evaluate_labeled import edit_distance
from final_ocr_dataset import ROOT, audit_candidates, check_protocol, conflicts, read
from yolo_character_dataset import validate
from yolo_character_pipeline import MixClassifiers


def main():
    check_protocol(); ds=ROOT/'datasets/yolo_character_round1'; validate(ds)
    out=ROOT/'reports/accuracy_recovery';out.mkdir(exist_ok=False)
    arms={'baseline':(640,640,.15),'plate960':(960,640,.15),'plate1280':(1280,640,.15),
          'char800':(640,800,.15),'char960':(640,960,.15),'padding25':(640,640,.25)}
    (out/'protocol.json').write_text(json.dumps({'arms':arms,'pixel_variants':VARIANTS,'selection':'development only; no promotion','scoring':'IoU>=.5, one-to-one'},indent=2))
    samples=[x['review'] for x in read(ds/'manifest.json')['rows'] if x['review']['split']=='val']
    models=MixClassifiers(); records=[]; traces=[]; a=audit_candidates(write=False)
    with Worker() as v1,worker() as client:
        for row in samples:
            im=cv2.imread(str(ROOT/row['image_path'])); x,y,w,h=row['plate_bbox'];gt=[[x+gx,y+gy,gw,gh] for gx,gy,gw,gh in [c['bbox'] for c in row['characters']]]
            old=run(im,v1,models,'all');
            for arm,(ps,cs,pad) in arms.items():
                r,_=analyze(im,client,models,ps,cs,pad,variants=VARIANTS if arm=='baseline' else ('raw',))
                box=r['plate']['crop_box'] if r['plate'] else None
                pred={'glyph_boxes':[g['bbox'] for g in r['glyphs']],'text':r['runs']['raw']}
                metrics=score(pred,box,gt,row); from segmentation_benchmark import match_boxes
                rec={**metrics,'arm':arm,'image':row['image_path'],'plate':match_boxes([p['bbox'] for p in r['plate_rows']],[row['plate_bbox']]),
                     'seconds':r['seconds'],'peak':r['peak_reserved'],'cold':r['cold'],'uncertain':r['uncertain_positions'],
                     'variants':{m:t['cnn'] for m,t in r['runs'].items()},'candidate':r['candidate']}
                if arm=='baseline':rec['matches_v1']=old['plate']==r['plate'] and old['glyph_boxes']==pred['glyph_boxes'] and old['text']==pred['text']
                records.append(rec)
        for f in sorted((ROOT/'data/alpr_failure_cases').glob('*/case.json')):
            paths=list(f.parent.glob('original.*'))
            if len(paths)!=1:continue
            path=paths[0];d,p,_=fingerprint(path)
            if d!=read(f)['sha256'] or p is None or conflicts(d,p,None,a['rows']+[r for r in a['known'] if r['split']=='test']):
                traces.append({'case':f.parent.name,'status':'skipped due to provenance or hash'});continue
            im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR); old=run(im,v1,models,'all')
            r,rois=analyze(im,client,models,variants=VARIANTS)
            dst=out/f.parent.name;dst.mkdir();cv2.imwrite(str(dst/'original.png'),im)
            if old['plate']:
                x,y,w,h=old['plate']['crop_box'];crop=im[y:y+h,x:x+w]
                cv2.imwrite(str(dst/'crop.png'),crop); overlay=crop.copy(); overview=im.copy()
                x0,y0,w0,h0=old['plate']['bbox'];cv2.rectangle(overview,(x0,y0),(x0+w0,y0+h0),(0,255,0),2)
                cv2.imwrite(str(dst/'plate.png'),overview)
                for i,(a0,b,c,e) in enumerate(old['glyph_boxes']):
                    roi=crop[b:b+e,a0:a0+c];cv2.imwrite(str(dst/f'{i:02}_roi.png'),roi)
                    cv2.imwrite(str(dst/f'{i:02}_gray.png'),cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY))
                    cv2.imwrite(str(dst/f'{i:02}_normalized.png'),pixels(roi,'raw'))
                    cv2.rectangle(overlay,(a0,b),(a0+c,b+e),(0,255,0),1);cv2.putText(overlay,str(i+1),(a0,max(8,b)),0,.3,(0,0,255),1)
                cv2.imwrite(str(dst/'glyph_order.png'),overlay)
            with Image.open(path) as pil:exif=pil.getexif().get(274); size=pil.size
            trace={'case':f.parent.name,'filename':read(f)['filename'],'exif_orientation':exif,'original_wh':size,
                   'decoded_hw':im.shape[:2],'pixel_range':[int(im.min()),int(im.max())],
                   'user_text_unreviewed':read(f).get('user_text_unreviewed'),'v1':old,'experimental':r}
            (dst/'trace.json').write_text(json.dumps(trace,indent=2));traces.append(trace)
    summary={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm]
        for name in (['raw','clahe','gamma','sharpen','adaptive','candidate'] if arm=='baseline' else ['raw']):
            refs=[r['reference'] for r in rows];pred=[r['candidate'] if name=='candidate' else r['variants'].get(name,r['variants']['raw']) for r in rows]
            base=[r['variants']['raw'] for r in records if r['arm']=='baseline']; summary[f'{arm}:{name}']={
                'exact':sum(x==y for x,y in zip(refs,pred)),'cer':sum(edit_distance(x,y) for x,y in zip(refs,pred))/sum(map(len,refs)),
                'fixed':sum(a!=b and a==c for a,b,c in zip(refs,base,pred)),'regressions':sum(a==b and a!=c for a,b,c in zip(refs,base,pred)),
                'plate':{k:sum(r['plate'][k] for r in rows) for k in ('tp','fp','fn')},'glyph':{k:sum(r['geometry'][k] for r in rows) for k in ('tp','fp','fn')},
                'median_seconds':float(np.median([r['seconds'] for r in rows if not r['cold']])), 'peak_reserved':max(r['peak'] for r in rows),
                'unstable_plates':sum(bool(r['uncertain']) for r in rows)}
    (out/'benchmark.json').write_text(json.dumps({'summary':summary,'records':records,'replays':traces},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()