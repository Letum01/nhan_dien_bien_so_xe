"""Summarize a frozen audit export, contact sheets and train-only review queue."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from annotation_store import AnnotationStore
from glyph_review import ROOT,CLASSES,plates_from_detection,normalized_text


def summarize_folder(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    dataset=folder/'dataset'
    with (dataset/'characters.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    counts={split:Counter(r['class_label'] for r in rows if r['split']==split) for split in ('train','validation','test')}
    class_rows=[{'class':c,**{s:counts[s][c] for s in counts},'missing_train':counts['train'][c]==0,
                 'train_1_or_2':1<=counts['train'][c]<=2} for c in CLASSES]
    with (folder/'class_counts_by_split.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    warnings=[]
    for kind in ('original','normalized'):
        sheet=np.full((((len(rows)+8)//9)*118,9*108,3),245,np.uint8)
        for index,row in enumerate(rows):
            target=dataset/row['image_path'];original=target.parent/'originals'/target.name
            glyph=cv2.imread(str(original if kind=='original' else target))
            if glyph is None:raise ValueError(str(target))
            gh,gw=glyph.shape[:2];ratio=min(94/gw,80/gh)
            resized=cv2.resize(glyph,(max(1,int(gw*ratio)),max(1,int(gh*ratio))))
            y=(index//9)*118;x=(index%9)*108
            sheet[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
            caption=f"{index}:{row['class_label']} {row['split'][:3]}"
            cv2.putText(sheet,caption,(x+2,y+101),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            if kind=='original':
                gray=cv2.cvtColor(glyph,cv2.COLOR_BGR2GRAY)
                _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                edges=[side for side in (binary[0],binary[-1],binary[:,0],binary[:,-1])]
                if any(np.mean(side>0)>.15 for side in edges):
                    warnings.append({'contact_index':index,'plate_id':row['plate_id'],'glyph_file':row['image_path'],
                                     'label':row['class_label'],'reason':'foreground touches boundary; heuristic NOT proven lost stroke'})
        cv2.imwrite(str(folder/f'ALL_GLYPHS_{kind}.jpg'),sheet)
    # Use snapshot parent labels, never predictions, to prioritize new train review.
    detection=AnnotationStore(folder/'annotations.sqlite3');ocr=AnnotationStore(folder/'ocr_glyphs.sqlite3')
    done={r['image_id'] for r in ocr.all_latest().values() if r.get('review_status')=='reviewed'}
    candidates=[r for r in plates_from_detection(detection) if r['split']=='train' and r['image_id'] not in done]
    def priority(r):
        labels=set(normalized_text(r['plate_text'])) if r['text_confirmed'] else set()
        return (not r['text_confirmed'],-sum(counts['train'][c]==0 for c in labels),-sum(counts['train'][c]<=2 for c in labels),r['image_path'])
    candidates.sort(key=priority)
    queue=[];groups=set()
    for row in candidates:
        if row['source_group'] in groups:continue
        groups.add(row['source_group'])
        queue.append({'image_path':row['image_path'],'plate_id':row['image_id'],'source_group':row['source_group'],
                      'split':'train','existing_human_text':row['plate_text'] if row['text_confirmed'] else '',
                      'text_confirmed':row['text_confirmed'],'glyph_status':'NOT_REVIEWED',
                      'reason':'existing confirmed text covers sparse train classes' if row['text_confirmed'] else 'different source; confirm text first, no inferred labels'})
        if len(queue)==15:break
    summary={'reviewed_plates':report['actual_glyph_reviewed_plates'],'glyphs':len(rows),
             'by_split':{s:sum(r['split']==s for r in rows) for s in counts},'class_counts':class_rows,
             'missing_train':[c for c in CLASSES if counts['train'][c]==0],
             'train_1_or_2':[c for c in CLASSES if 1<=counts['train'][c]<=2],
             'only_validation':[c for c in CLASSES if not counts['train'][c] and counts['validation'][c]],
             'boundary_warnings':warnings,'next_train_review':queue,
             'label_validation':report['export_validator'],'training':'NOT_RUN',
             'milestone_audit_record':ocr.preference('milestone10_audit'),
             'warning':'All contact flags require human inspection; no auto-label correction or training. Test has no OCR glyphs.'}
    (folder/'milestone_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Audit mốc 10 biển OCR',f"\nReviewed {summary['reviewed_plates']}; glyph {len(rows)}; splits {summary['by_split']}.",
           f"\nValidator {report['export_validator']['status']}; parent/sequence checks: "+str([(Path(r['image']).name,r['status'],r['issues']) for r in report['details']]),
           '\n|Class|Train|Validation|Test|','|---|---:|---:|---:|']
    lines += [f"|{r['class']}|{r['train']}|{r['validation']}|{r['test']}|" for r in class_rows]
    lines += [f"\nMissing train: {summary['missing_train']}",f"\nTrain 1–2 samples: {summary['train_1_or_2']}",
              f"\nOnly validation: {summary['only_validation']}",
              f"\nBoundary warnings {len(warnings)} (heuristic, not confirmed errors). See milestone_summary.json for exact contact indices.",
              '\n## Next 15 train plates (not labeled by audit)']
    lines += [f"- {r['image_path']} — {r['source_group']} — text confirmed={r['text_confirmed']}" for r in queue]
    lines += ['\nALL_GLYPHS_original.jpg and ALL_GLYPHS_normalized.jpg include all glyphs with index/class/split. '
              'No classifier training. Three train plates are insufficient, many classes missing train and no OCR test. '
              'Numeric PASS is not a new human review of pixel quality.']
    (folder/'MILESTONE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SUMMARY',folder/'MILESTONE_REPORT.md');print('missing train',summary['missing_train']);print('low',summary['train_1_or_2'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    summarize_folder(args.audit)