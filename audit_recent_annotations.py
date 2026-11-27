"""Read-only SQLite snapshot audit; draft labels NEVER promoted for export."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2

from dataset_audit import fingerprint
from dataset_validation import validate_boxes, check_split_leakage
from label_detection import validate_annotation, export_yolo
from validate_yolo_dataset import validate_dataset

ROOT = Path(__file__).resolve().parent


def inspect_payload(payload, origin, base_version=None):
    row = dict(image_path=payload.get('image_path',''), image_id=payload.get('image_id'),
               origin=origin, review_status=payload.get('review_status','unlabeled'),
               annotation_version=payload.get('annotation_version',base_version),
               source_group=payload.get('source_group',''), split=payload.get('split'),
               boxes=payload.get('boxes',[]), confirmed_by_user=payload.get('confirmed_by_user',False),
               issues=[], geometry_status='NOT_TESTED', text=[], glyph_count=0)
    path=ROOT/row['image_path']
    if not path.is_file():
        row['issues'].append('Source image missing');row['status']='FAIL';return row
    sha,phash,shape=fingerprint(path)
    row.update(sha256=sha,phash=f'{phash:016x}' if phash is not None else None,
               resolution=[shape[1],shape[0]] if shape else None)
    if shape is None:
        row['issues'].append('Image cannot be decoded');row['status']='FAIL';return row
    if sha!=row['image_id']:row['issues'].append('image_id does not match source bytes')
    try:
        validate_boxes(row['boxes'],shape[1],shape[0]);row['geometry_status']='PASS'
    except (ValueError,TypeError) as exc:
        row['geometry_status']='FAIL';row['issues'].append(str(exc))
    row['box_count']=len(row['boxes'])
    if not row['boxes'] and not (origin=='reviewed_revision' and row['confirmed_by_user']):
        row['issues'].append('No boxes; negative-image status not explicitly reviewed')
    for i,box in enumerate(row['boxes']):
        if len(box)==4 and (box[2]<8 or box[3]<8):
            row['issues'].append(f'Box {i} suspiciously small ({box[2]}x{box[3]}); inspect/delete manually')
    if not row['source_group'].strip():row['issues'].append('Missing source_group')
    if row['split'] not in ('train','val','test'):row['issues'].append('Split missing/UNASSIGNED')
    elif origin!='reviewed_revision':row['issues'].append('Split is draft metadata only, not verified')
    if origin!='reviewed_revision':row['issues'].append('Draft only: no reviewed revision saved')
    if not row['confirmed_by_user']:row['issues'].append('No explicit human detection confirmation')
    plates=payload.get('plates',[])
    if len(plates)!=len(row['boxes']):row['issues'].append('Plate metadata count differs from box count')
    for i,plate in enumerate(plates):
        row['text'].append({'plate_index':i,'raw_text':plate.get('plate_text',''),
                            'confirmed':plate.get('text_confirmed',False),'unreadable':plate.get('unreadable'),
                            'layout':plate.get('layout'),'glyph_count':len(plate.get('characters',[]))})
        chars=plate.get('characters',[]);row['glyph_count']+=len(chars)
        if chars and i<len(row['boxes']):
            try:validate_boxes([char['bbox'] for char in chars],row['boxes'][i][2],row['boxes'][i][3])
            except (ValueError,KeyError,TypeError) as exc:row['issues'].append(f'Glyph geometry {i}: {exc}')
            if any(char.get('label') not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' or not char.get('confirmed') for char in chars):
                row['issues'].append(f'Plate {i} contains unconfirmed/invalid glyph labels')
    row['status']='FAIL' if row['geometry_status']=='FAIL' or sha!=row['image_id'] else 'NEEDS_REVIEW' if row['issues'] else 'PASS'
    row['ocr_ready']=bool(plates) and all(p.get('plate_text','').strip() and p.get('text_confirmed') and not p.get('unreadable') for p in plates)
    row['detection_export_eligible']=False
    if origin=='reviewed_revision':
        try:validate_annotation(payload);row['detection_export_eligible']=True
        except ValueError as exc:row['issues'].append(f'Detection validator: {exc}');row['status']='FAIL'
    return row


def main():
    database=ROOT/'data/annotations.sqlite3'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=ROOT/'reports'/f'annotation_review_{stamp}';folder.mkdir(parents=True,exist_ok=False)
    # SQLite backup API gives one consistent snapshot even if UI is running.
    source=sqlite3.connect(f'file:{database.as_posix()}?mode=ro',uri=True)
    snapshot=sqlite3.connect(':memory:');source.backup(snapshot);source.close()
    revisions=[json.loads(r[0]) for r in snapshot.execute('SELECT r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id)t ON r.image_id=t.image_id AND r.version=t.v')]
    drafts=[(json.loads(p),v) for p,v in snapshot.execute('SELECT payload,base_version FROM drafts')]
    reviewed=[row for row in revisions if row.get('review_status')=='reviewed']
    rows=[inspect_payload(r,'reviewed_revision' if r.get('review_status')=='reviewed' else 'other_revision') for r in revisions]
    rows += [inspect_payload(p,'draft',v) for p,v in drafts]
    # Visual evidence for human follow-up; annotate source copies, never originals.
    previews=folder/'box_previews';previews.mkdir()
    for index,row in enumerate(rows):
        image=cv2.imread(str(ROOT/row['image_path']))
        if image is None:continue
        for bi,box in enumerate(row['boxes']):
            if len(box)!=4:continue
            x,y,w,h=map(int,box)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,180,255),2)
            cv2.putText(image,str(bi),(x,max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        name=f'{index:02d}_{Path(row["image_path"]).stem}.jpg'
        cv2.imwrite(str(previews/name),image)
        row['preview']=str((previews/name).relative_to(folder))
    duplicate_pairs=[];group_conflicts=[]
    for i,a in enumerate(rows):
        for b in rows[:i]:
            if a['image_id']==b['image_id']:continue
            same=a.get('sha256') and a.get('sha256')==b.get('sha256')
            distance=(int(a['phash'],16)^int(b['phash'],16)).bit_count() if a.get('phash') and b.get('phash') else None
            if same or (distance is not None and distance<=6):
                duplicate_pairs.append({'a':a['image_path'],'b':b['image_path'],'exact':bool(same),'phash_distance':distance,
                                        'same_group':bool(a['source_group']) and a['source_group']==b['source_group'],
                                        'splits':[a['split'],b['split']],'review':'candidate_only'})
            if a['source_group'] and a['source_group']==b['source_group'] and a['split']!=b['split']:
                group_conflicts.append({'a':a['image_path'],'b':b['image_path'],'group':a['source_group'],'splits':[a['split'],b['split']]})
    external_pairs=[]
    audit_path=ROOT/'reports/dataset_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        paths={str(Path(r['image_path'])) for r in rows}
        external_pairs=[pair for pair in audit.get('near_duplicate_candidates',[]) if str(Path(pair['a'])) in paths or str(Path(pair['b'])) in paths]
    leakage={'status':'BLOCKED' if not reviewed else 'PASS','reason':'No reviewed samples' if not reviewed else None}
    if reviewed:
        try:check_split_leakage(reviewed,ROOT)
        except ValueError as exc:leakage={'status':'FAIL','reason':str(exc)}
    # Export only validated reviewed rows, never substitute draft payloads.
    eligible=[r for r in reviewed if next(x for x in rows if x['image_id']==r['image_id'] and x['origin']=='reviewed_revision')['detection_export_eligible']]
    manifest=folder/'reviewed_export_input.json';manifest.write_text(json.dumps(eligible,ensure_ascii=False,indent=2),encoding='utf-8')
    destination=folder/'yolo_trial'
    try:
        if len(eligible)!=len(reviewed):
            raise ValueError("Reviewed samples failed validation; refusing to silently remove images/change test set")
        export_yolo(manifest,destination)
        exported={'status':'PASS','destination':str(destination)}
    except ValueError as exc:exported={'status':'BLOCKED','reason':str(exc),'destination_created':destination.exists()}
    try:validation=validate_dataset(destination)
    except ValueError as exc:validation={'status':'BLOCKED' if not destination.exists() else 'FAIL','reason':str(exc)}
    report={'database':str(database),'snapshot_utc':stamp,'total_revision_records':snapshot.execute('SELECT COUNT(*) FROM revisions').fetchone()[0],
            'latest_revision_images':len(revisions),'actual_reviewed_images':len(reviewed),'draft_images':len(drafts),
            'counts':dict(Counter(r['status'] for r in rows)),'draft_states':dict(Counter(p.get('review_status') for p,_ in drafts)),
            'draft_splits_unverified':dict(Counter(p.get('split') for p,_ in drafts)),
            'rows':rows,'duplicates_within_saved_rows':duplicate_pairs,'near_duplicate_candidates_across_source_dataset':external_pairs,
            'source_group_conflicts':group_conflicts,'reviewed_split_leakage':leakage,'export':exported,'yolo_validator':validation,
            'training_status':'BLOCKED' if exported['status']!='PASS' else 'NOT_AUTHORIZED_pending_sample_quality_and_training_checks',
            'note':'Geometry PASS means numeric validity, not visual proof of correct plate. Text/glyph not required for YOLO, required separately for OCR. No annotation states modified.'}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['image_path','origin','status','review_status','box_count','boxes','geometry_status','source_group','split','ocr_ready','glyph_count','issues']
    with (folder/'report.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Kiểm tra nhãn SQLite',f'\nReviewed thực sự: **{len(reviewed)}**. Draft: **{len(drafts)}**.',
           f'\nKết quả: {report["counts"]}. Export: {exported}. Validator: {validation}.',
           '\nKhông thay đổi nhãn. Cần xác nhận reviewed rõ ràng sau khi kiểm tra source_group/split/box. Text/glyph trống không chặn YOLO nhưng chưa đủ OCR.\n',
           '|Ảnh|Trạng thái|Box|Group|Split|Cần sửa|','|---|---|---|---|---|---|']
    for r in rows:lines.append(f"|{r['image_path']}|{r['status']}|{r.get('box_count',0)}|{r['source_group'] or '(trống)'}|{r['split']}|{'; '.join(r['issues'])}|")
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['actual_reviewed_images','draft_images','counts','draft_states','draft_splits_unverified','export','yolo_validator']},ensure_ascii=False,indent=2))
    print('REPORT',folder)


if __name__=='__main__':main()