"""Frozen35plate summary and new-hard-case contact sheets; no model training."""
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from annotation_store import AnnotationStore
from glyph_review import ROOT,CLASSES
from ocr_hard_review import load_hard_queue

AUDIT=ROOT/'reports/glyph_audit_20261125T095817713548Z'


def main():
    ocr=AnnotationStore(AUDIT/'ocr_glyphs.sqlite3');detection=AnnotationStore(AUDIT/'annotations.sqlite3')
    hard=load_hard_queue(detection,ocr);completed=hard['completed']
    manifest=list(csv.DictReader((AUDIT/'dataset/characters.csv').open(encoding='utf-8-sig')))
    ids={r['image_id'] for r in completed};glyphs=[r for r in manifest if r['plate_id'] in ids]
    mapping={r['image_id']:r for r in completed}
    warnings=[]
    for kind in ('raw','normalized'):
        sheet=np.full((((len(glyphs)+8)//9)*116,9*108,3),240,np.uint8)
        for i,row in enumerate(glyphs):
            path=AUDIT/'dataset'/row['image_path']
            if kind=='raw':path=path.parent/'originals'/path.name
            image=cv2.imread(str(path));h,w=image.shape[:2];scale=min(98/w,80/h)
            image=cv2.resize(image,(max(1,int(w*scale)),max(1,int(h*scale))))
            x=i%9*108;y=i//9*116;sheet[y:y+image.shape[0],x:x+image.shape[1]]=image
            cv2.putText(sheet,f"{i}:{row['class_label']}",(x,y+102),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            if kind=='raw':
                original=cv2.imread(str(path),0)
                _,binary=cv2.threshold(original,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                edges=[label for label,edge in [('top',binary[0]),('bottom',binary[-1]),('left',binary[:,0]),('right',binary[:,-1])] if np.mean(edge>0)>.15]
                if edges:warnings.append({'sheet_index':i,'image':mapping[row['plate_id']]['image_path'],
                   'glyph_index':int(Path(row['image_path']).stem),'label':row['class_label'],'bbox':row['bbox'],'edges':edges,
                   'reason':'tight foreground boundary screening, not confirmed cutoff'})
        cv2.imwrite(str(AUDIT/f'NEW_HARD_83_{kind}.jpg'),sheet)
    counts={split:Counter(row['class_label'] for row in manifest if row['split']==split) for split in ('train','validation','test')}
    classes=[{'class':c,**{s:counts[s][c] for s in counts}} for c in CLASSES]
    old=json.loads((ROOT/'reports/crop_padding_20261125T084749402271Z/results.json').read_text(encoding='utf-8'))
    old_ids={r['plate_id'] for r in old['records']}
    common=[r['image_path'] for r in completed if r['image_id'] in old_ids]
    next5=hard['pending'][:5]
    summary={'reviewed_plates':35,'glyphs':287,'train':230,'validation':57,'test':0,
        'hard_completed':len(completed),'hard_pending':len(hard['pending']),'new_hard_glyphs':len(glyphs),
        'class_counts':classes,'missing_train':[c for c in CLASSES if not counts['train'][c]],
        'train_1_or_2':[c for c in CLASSES if 1<=counts['train'][c]<=2],
        'new_hard_images':[r['image_path'] for r in completed],'historical_benchmark_overlap':common,
        'warnings':warnings,'next5':[{'image':r['image_path'],'group':r['source_group'],'split':r['split'],'priority':r['hard_priority']} for r in next5],
        'counter_root_cause':'Multiple latest reads during one render + no rerun after save; render can combine N and N-1 snapshots. No cached summary function.',
        'counter_fix':'One reviewed snapshot supplied to progress and hard queue + st.rerun after reviewed save.',
        'root_cause_limit':'Original mismatching browser frame not available; code permits and explains observed 8-glyph delta, not a recovered historical event trace.',
        'benchmark_readiness':'READY train-development expansion on10newhardplates83glyphs, NOT independent holdout; after tuning need new held-out sources.',
        'training':'NOT_RUN','ground_truth_changed':False}
    (AUDIT/'hard35_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Audit mốc35biển OCR và bộ đếm',
       '\nSQLite actual:61Detectionreviewed,35Glyphreviewed/287confirmed;28trainplates230glyphs,7valplates57glyphs,test0. '
       '10hardcases83newglyphscompleted,18pending. No label changes.',
       '\n## Counter mismatch',
       'Hard230 is correct;222train+57val=279 is one8glyphtrainplatebehind. UI called all_latest independently '
       'for header/progress/hard, then saved reviewed at bottom without rerun. No cached aggregate. '
       'Concurrent save/rerun windows could mix snapshots. Original stale browser frame not retained, so cannot '
       'recover exact event timeline. Fixed shared reviewed snapshot and immediate rerun after save, no data edits. '
       'Chromium syntheticfixture actualreview save→immediateheader1plate1glyph→reloadsamePASS. '
       'Tests ensure progress does not reread snapshot. No realreview labels fabricated.',
       '\n## Audit/export',
       'Automatic hard10 already ran:reports/glyph_audit_20261125T095217018567Z,validatorPASS287. '
       'New manualaudit/export:'+str(AUDIT.relative_to(ROOT))+',all35parent/sequencechecksPASS,leakagePASS. '
       'This does not prove visual pixel quality; source splits unchanged.',
       '\n## Class counts', '|Class|Train|Val|Test|','|---|---:|---:|---:|']
    lines += [f"|{r['class']}|{r['train']}|{r['validation']}|{r['test']}|" for r in classes]
    lines += [f"\nMissingtrain:{summary['missing_train']}.1–2sampletrain:{summary['train_1_or_2']}.",
       '\n## Hard cases / quality',
       'NEW_HARD_83_raw.jpg and normalized.jpg show all83newglyphs. Perplatecontact/overlay also available. '
       f"{len(warnings)}boundaryheuristicwarnings,NOTconfirmederrors; exactimage/glyphindex/bbox inhard35_summary.json. "
       "Tightcropsnaturallytouchinkboundary; inspect source/normalized before correction. No automaticpadding/relabel/rejection.",
       f'\nHistorical25platebenchmarkoverlap:{common}. Most newhard sources were absent from prior reviewedglyphbenchmark. '
       'They represent small/blurred/frame conditions by queue proxies, not new human difficulty labels. '
       'READY for a separate TRAINdevelopment benchmark on10newplates,freezeprotocol first; not independent finaltest '
       'and do not tune on7validation. No classifier training/segmentationpromotion.',
       '\n## Next5 hard review suggestions (existingqueueorder)']
    lines += [f"- {Path(r['image_path']).name} / {r['source_group']} / {r['split']}" for r in next5]
    lines += ['\n## Tests','128pytestPASS;datasetvalidator287PASS;Chromiumsave/reloadcounterPASSfixture. '
              'No model/threshold/productionsegmentationchanges. ExistingOCRtestabsent.']
    (ROOT/'reports/OCR_HARD_MILESTONE_35.md').write_text('\n'.join(lines),encoding='utf-8')
    print('report',ROOT/'reports/OCR_HARD_MILESTONE_35.md','hard glyphs',len(glyphs),'warnings',len(warnings),'overlap',common)


if __name__=='__main__':main()