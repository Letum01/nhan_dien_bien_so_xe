"""Summarize frozen diagnostic artifacts without editing model/annotation data."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'reports/pilot_confidence_analysis_v3'
    report=json.loads((folder/'analysis.json').read_text(encoding='utf-8'))
    records=[]
    for row in report['label_audit']:
        for index,box in enumerate(row['boxes']):
            records.append({'image':row['image'],'split':row['split'],'box_index':index,
                            'xywh':box['xywh'],'xyxy':box['xyxy'],
                            'area_fraction':box['area_fraction'],'aspect_ratio':box['aspect_ratio'],
                            'flags':box['flags'],'export_matches_xywh':row['export_matches_xywh'],
                            'screening_status':row['screening_status'],'overlay':row['overlay']})
    with (folder/'bbox_screening.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    plan=json.loads((ROOT/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    frozen={str(Path(row['image_path'])) for row in plan['frozen_pilot']}
    candidates={str(Path(row['image_path'])) for row in plan['candidates']}
    forbidden={str(Path(path)) for path in plan['excluded_pilot_related_paths']}
    assert not candidates & forbidden
    assert not candidates & frozen
    assert len(candidates)==120
    assert all(row['review_status']=='unlabeled' and row['split']=='UNASSIGNED' for row in plan['candidates'])
    summary={'status':'BLOCKED_FOR_TRAINING','label_count':len(report['label_audit']),
             'bbox_screening_flags':sum(bool(row['flags']) for row in records),
             'export_coordinate_match':all(row['export_matches_xywh'] for row in records),
             'visual_verification':'16.jpg overlay inspected; remaining overlays supplied for human inspection, not auto-approved',
             'new_candidates':len(candidates),'target_total':141,
             'candidate_pilot_duplicate_exclusion':'PASS','test_membership_frozen':plan['frozen_test_images'],
             'model_or_gt_modified':False,'product_threshold_modified':False,
             'needed':['Human review/annotation of expanded training and validation sources',
                       'Manual verification of suspicious visual labels if any',
                       'Frozen test integrity and fresh snapshot validator PASS'],
             'config':'configs/yolo_round2.json'}
    (ROOT/'reports/round2_plan/readiness.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()