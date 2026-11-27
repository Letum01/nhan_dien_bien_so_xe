"""Export per-match confusion/error rows and before/after per-plate CSVs."""
import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix


def main(folder):
    folder=Path(folder);report=json.loads((folder/'benchmark.json').read_text(encoding='utf-8'))
    selected=report['selected_variant_train_only']
    records=[];matched=[];summary={}
    for row in report['per_plate']:
        if row['variant'] not in ('baseline',selected):continue
        score=row['segmentation']
        # Preserve NOT TESTED instead of confusing zero with classification failure.
        for pair in row.get('matched_classification',[]):
            matched.append({'image':row['image'],'plate_id':row['plate_id'],'split':row['split'],'variant':row['variant'],**pair})
        records.append({'image':row['image'],'split':row['split'],'variant':row['variant'],
                        'tp':score['tp'],'fp':score['fp'],'fn':score['fn'],
                        'same_count':score['same_count'],'all_matched':score['full_match'],
                        'order_all_correct':score['complete_order_correct'],'text':row.get('raw_text'),
                        'reviewed_text':row.get('reviewed_text'),'exact_text':row.get('exact_text'),
                        'split_overlap_candidates':score['split_overlap_candidates'],
                        'merge_overlap_candidates':score['merge_overlap_candidates']})
    for split in ('train','val'):
        for variant in ('baseline',selected):
            pairs=[m for m in matched if m['split']==split and m['variant']==variant]
            classes=list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            summary[split+'_'+variant]={'matched_samples':len(pairs),
              'correct':sum(m['prediction']==m['expected'] for m in pairs),
              'class_order':classes,'confusion_matrix':confusion_matrix([m['expected'] for m in pairs],[m['prediction'] for m in pairs],labels=classes).tolist()}
    for name,data in [('per_plate_error_breakdown.csv',records),('matched_glyph_predictions.csv',matched)]:
        if data:
            with (folder/name).open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (folder/'matched_classifier_confusion.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    changes=[]
    for base in [r for r in records if r['variant']=='baseline']:
        new=next(r for r in records if r['image']==base['image'] and r['variant']==selected)
        def f1(r):return 2*r['tp']/max(1,2*r['tp']+r['fp']+r['fn'])
        changes.append({'image':base['image'],'split':base['split'],
                        'status':'improved' if f1(new)>f1(base) else 'regressed' if f1(new)<f1(base) else 'same'})
    (folder/'improvements_and_regressions.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
    import cv2
    root=Path(__file__).resolve().parent
    e2e=folder/'end_to_end_overlays';e2e.mkdir(exist_ok=True)
    source_rows=json.loads((Path(report['protocol']['snapshot'])/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    for index,item in enumerate(report['yolo_end_to_end']):
        row=next(r for r in source_rows if r['image_path']==item['image'])
        image=cv2.imread(str(root/item['image']))
        gx,gy,gw,gh=row['plate_bbox'];cv2.rectangle(image,(gx,gy),(gx+gw,gy+gh),(0,255,0),2)
        for detection in item['detector_boxes']:
            x,y,w,h=detection['bbox'];cv2.rectangle(image,(x,y),(x+w,y+h),(0,180,255),2)
            cv2.putText(image,f"YOLO {detection['score']:.3f}",(x,max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,255),1)
        cv2.putText(image,item['prediction'],(5,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        cv2.imwrite(str(e2e/f'{index:02d}_{Path(item["image"]).stem}.jpg'),image)
    rates={split:{variant:{'equal_glyph_count_rate':m['same_count_plates']/m['plates'],
                            'full_one_to_one_match_rate':m['full_match_plates']/m['plates'],
                            'full_reading_order_rate':m['complete_order_correct_plates']/m['plates'],
                            'matched_classifier_accuracy':m['glyph_match_classifier_correct']/m['glyph_match_classifier_samples'] if m['glyph_match_classifier_samples'] else None,
                            'classification_status':'evaluated' if variant in ('baseline',selected) else 'NOT_TESTED'}
                  for variant,m in variants.items()} for split,variants in report['summary'].items()}
    (folder/'rates.json').write_text(json.dumps(rates,indent=2),encoding='utf-8')
    print('Changes',[(r['split'],Path(r['image']).name,r['status']) for r in changes if r['status']!='same'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)