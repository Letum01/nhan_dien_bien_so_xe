"""Manual-only test corpus, deliberately separate from both training stores."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from dataset_audit import fingerprint
from dataset_validation import validate_boxes
from glyph_review import ROOT, plates_from_detection, validate_glyph_plate
from yolo_character_dataset import ReadOnlyStore, sha

PROTOCOL=ROOT/'reports/final_ocr_protocol.json'
PROTOCOL_HASH='865ca7d977b0615da890303628d4c23cbad6c48063d829538a7f892a3e22d59a'
AUDIT=ROOT/'reports/final_ocr_candidate_audit.json'
STORE=ROOT/'data/final_ocr_test.sqlite3'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_protocol():
    if sha(PROTOCOL)!=PROTOCOL_HASH:
        raise ValueError('Frozen evaluation protocol changed')
    for path,digest in read(PROTOCOL)['hashes'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError(f'Frozen dependency changed: {path}')


def relative_image(value):
    path=ROOT/Path(value)
    try:
        return path.resolve().relative_to(ROOT).as_posix().casefold()
    except ValueError:
        return path.resolve().as_posix().casefold()


def conflicts(digest, phash, group, others):
    return [r['image'] for r in others if digest==r['sha256'] or
            (phash^int(r['phash'],16)).bit_count()<=6 or
            (group and group.casefold()==r.get('source_group','').strip().casefold())]


def final_reference(row):
    return {'image':row['image_path'],'sha256':row['source_sha256'],'phash':row['source_phash'],
            'source_group':row['source_group'],'split':'test','final_peer':True}


def check_final_conflicts(entry, group, references):
    """Same-source final peers are allowed, duplicate files are not."""
    messages=[]; seen=set()
    for r in references:
        image=relative_image(r['image']); peer=r.get('final_peer') is True and r.get('split')=='test'
        # A review of the same final item is not another sample. An exposure
        # reference, even at the same path, must still fail the leakage gate.
        if peer and image==relative_image(entry['image']) and r['sha256']==entry['sha256']:continue
        dist=(int(entry['phash'],16)^int(r['phash'],16)).bit_count()
        group_match=bool(group and group.strip().casefold()==r.get('source_group','').strip().casefold())
        reasons=[]
        if entry['sha256']==r['sha256']:reasons.append('SHA-256 trùng')
        if dist<=6:reasons.append(f'pHash gần trùng ({dist} bit, ngưỡng ≤6)')
        if group_match and not peer:reasons.append('source_group trùng nguồn tham chiếu')
        if not reasons:continue
        split=r.get('split') or 'không rõ (lịch sử sử dụng)'
        msg=f"{', '.join(reasons)}: {r.get('filename') or Path(r['image']).name} — split: {split}, nhóm: {r.get('source_group') or 'chưa rõ'}."
        if msg not in seen:messages.append(msg);seen.add(msg)
    if messages:
        raise ValueError(' '.join(messages)+' Kiểm tra ảnh/nguồn và dùng ảnh độc lập khác nếu trùng; không đổi group chỉ để vượt kiểm tra.')


def exposure_index(source_root=ROOT):
    source_root=Path(source_root).resolve()
    known={}
    def add(path, label):
        key=relative_image(path)
        if key and key.startswith('dataset/license_plate_dataset/'):
            known.setdefault(key,set()).add(label)
    detection=ReadOnlyStore(source_root/'data/annotations.sqlite3').all_latest().values()
    groups={relative_image(r['image_path']):r.get('source_group','') or '' for r in detection}
    splits={relative_image(r['image_path']):r['split'] for r in detection}
    for r in detection: add(r['image_path'],'detection_registry:'+r['split'])
    with (source_root/'reports/round2_plan/candidates.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f): add(row['image_path'],'round2_review_queue')
    excluded={'dataset_audit.json','ocr_test_provenance.json','final_test_new10_audit.json',AUDIT.name,PROTOCOL.name}
    for folder in ('reports','experiments','debug_outputs'):
        for path in sorted((source_root/folder).rglob('*.json')):
            if path.name in excluded or path.name.startswith('final_ocr_corpus_'):continue
            try: stack=[read(path)]
            except (OSError,ValueError):continue
            while stack:
                value=stack.pop()
                if isinstance(value,dict):stack.extend(value.values())
                elif isinstance(value,list):stack.extend(value)
                elif isinstance(value,str) and 'license_plate_dataset' in value and value.lower().endswith(('.jpg','.jpeg','.png')):
                    p=Path(value)
                    add(Path('dataset')/p if p.parts and p.parts[0]=='license_plate_dataset' else p,path.relative_to(source_root).as_posix())
    index=[]
    for image,refs in sorted(known.items()):
        path=source_root/image
        if path.is_file():
            digest,p,shape=fingerprint(path)
            if shape:index.append({'image':image,'sha256':digest,'phash':f'{p:016x}','source_group':groups.get(image,''),'split':splits.get(image),'references':sorted(refs)})
    return index


def audit_candidates(store_path=STORE, *, write=True, source_root=ROOT):
    from final_ocr_intake import registered
    check_protocol()
    source_root=Path(source_root).resolve()
    known=exposure_index(source_root); rows=[]
    candidates=read(source_root/'reports/ocr_test_provenance.json')['candidates']+registered(store_path)
    for entry in candidates:
        image=entry['image']; path=source_root/image
        if path.is_file():digest,p,shape=fingerprint(path)
        else:digest,p,shape=None,None,None
        matches=conflicts(digest,p,None,known) if shape else []
        reasons=[]
        if not shape or digest!=entry['sha256'] or f'{p:016x}'!=entry['phash']:
            reasons.append('Missing, unreadable or fingerprint changed')
        if matches: reasons.append('Exact/pHash conflict with known exposure')
        rows.append({**entry,'source_group':'UNKNOWN','vehicle_capture_video':'UNKNOWN',
                     'status':'REPLACE' if reasons else 'UNKNOWN','reasons':reasons,
                     'exposure_matches':[r for r in known if r['image'] in matches]})
    for i,row in enumerate(rows):
        for other in rows[:i]:
            if conflicts(row['sha256'],int(row['phash'],16),None,[other]):
                row['status']='REPLACE';row['reasons'].append('Duplicate candidate: '+other['image'])
    replacements={r['replaces']:r['image'] for r in rows if r.get('replaces') and r['status']!='REPLACE'}
    for row in rows:
        row['superseded_by']=replacements.get(relative_image(row['image']))
    output={'rows':rows,'known_exposure_images':len(known),'known_groups':sorted({r['source_group'].casefold() for r in known if r['source_group']}),
            'limitations':'Indexed workspace evidence only; absence is not proof of independence. Source, vehicle/capture/video and external exposure need human confirmation.'}
    if write:AUDIT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    output['known']=known
    return output


def pilot_rows(detection=None,ocr=None):
    detection=detection or ReadOnlyStore(ROOT/'data/annotations.sqlite3')
    ocr=ocr or ReadOnlyStore(ROOT/'data/ocr_glyphs.sqlite3')
    history=ocr.all_latest()
    paths={relative_image(r['image']) for r in read(ROOT/'reports/ocr_test_provenance.json')['pilot_test']}
    rows=[]
    for parent in plates_from_detection(detection):
        if parent['split']!='test' or relative_image(parent['image_path']) not in paths:continue
        prior=history.get(parent['image_id'],{})
        if prior.get('review_status')=='reviewed' and all(prior[k]==parent[k] for k in ('detection_revision','plate_bbox','split','source_group')):
            parent={**parent,**prior,'original_ocr_revision':prior['annotation_version']}
        rows.append(parent)
    return sorted(rows,key=lambda r:r['image_path'])


def validate_manual(row, shape):
    if row['split']!='test' or row['cohort'] not in ('pilot','final') or not row.get('plate_box_confirmed'):
        raise ValueError('Confirm the manual test plate bbox')
    box=row['plate_bbox']
    from yolo_character_dataset import integer_boxes
    integer_boxes([box],shape[1],shape[0]); integer_boxes([c['bbox'] for c in row['characters']],box[2],box[3])
    validate_glyph_plate(row,box[2],box[3])
    if row['cohort']=='final':
        if (not row.get('provenance_confirmed') or row.get('confirmed_group')!=row['source_group'] or
                row.get('confirmed_evidence')!=row.get('source_evidence') or len(row.get('source_evidence','').strip())<20 or
                row['source_group'].strip().casefold() in ('','unknown','unverified','needs_human_confirmation')):
            raise ValueError('Confirm a real group and document capture/vehicle/history evidence')


def validate_corpus(rows, store_path=STORE):
    check_protocol()
    audit_result=audit_candidates(store_path,write=False)
    audit={relative_image(r['image']):r for r in audit_result['rows']}
    known=audit_result['known']; accepted=[]
    if not rows:raise ValueError('No reviewed independent final-test records')
    for row in rows:
        if row.get('review_status')!='reviewed' or not row.get('confirmed_by_user') or row.get('cohort')!='final':
            raise ValueError('Final corpus requires explicitly reviewed final records')
        entry=audit.get(relative_image(row['image_path']))
        if not entry or entry['status']=='REPLACE' or entry.get('superseded_by'):raise ValueError('Candidate missing, replaced or has confirmed conflict')
        digest,p,shape=fingerprint(ROOT/row['image_path'])
        if digest!=entry['sha256'] or digest!=row['source_sha256'] or f'{p:016x}'!=row['source_phash']:
            raise ValueError('Source fingerprint mismatch')
        validate_manual(row,shape)
        if any(relative_image(r['image'])==relative_image(row['image_path']) for r in accepted):
            raise ValueError('Duplicate corpus row')
        check_final_conflicts(entry,row['source_group'],known+accepted)
        accepted.append(final_reference(row))
    return {'plates':len(rows),'glyphs':sum(len(r['characters']) for r in rows),
            'groups':len({r['source_group'].strip().casefold() for r in accepted})}


def freeze(store_path=STORE):
    """Only after the operator explicitly invokes freeze, with all usable candidates resolved."""
    rows=[r for r in ReadOnlyStore(store_path).all_latest().values() if r.get('cohort')=='final']
    included=[r for r in rows if r.get('review_status')=='reviewed']
    dismissed={relative_image(r['image_path']) for r in rows if r.get('review_status')=='rejected' and r.get('exclusion_reason','').strip()}
    accepted={relative_image(r['image_path']) for r in included}
    audit=audit_candidates(store_path)
    pending=[e['image'] for e in audit['rows'] if e['status']!='REPLACE' and not e.get('superseded_by') and relative_image(e['image']) not in accepted|dismissed]
    if pending:raise ValueError(f'{len(pending)} candidates still pending; review or record a reason for exclusion')
    counts=validate_corpus(included,store_path)
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=ROOT/'reports'/f'final_ocr_corpus_{now}.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump({'version':now,'protocol_sha256':PROTOCOL_HASH,'counts':counts,'rows':included,
                   'human_exclusions':[r for r in rows if relative_image(r['image_path']) in dismissed],
                   'intake_audit':audit['rows']},f,ensure_ascii=False,indent=2)
    return path


def export_pilot(destination):
    """Separate immutable label manifest from existing human-reviewed pilot rows."""
    rows=pilot_rows()
    if len(rows)!=3 or any(not r.get('original_ocr_revision') for r in rows):
        raise ValueError('Expected three reviewed pilot plates')
    for row in rows:
        digest,p,shape=fingerprint(ROOT/row['image_path'])
        if not shape:raise ValueError('Unreadable pilot image')
        validate_manual({**row,'cohort':'pilot','plate_box_confirmed':True},shape)
        row.update(source_sha256=digest,source_phash=f'{p:016x}')
    with Path(destination).open('x',encoding='utf-8') as f:
        json.dump({'scope':'pilot_diagnostics_exposed_not_independent', 'split':'test','rows':rows,
                   'glyphs':sum(len(r['characters']) for r in rows)},f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','freeze','export-pilot'));parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.action=='audit':
        result=audit_candidates(); print('Candidates',len(result['rows']),'replace',sum(r['status']=='REPLACE' for r in result['rows']),'indexed',result['known_exposure_images'])
    elif args.action=='freeze':print(freeze())
    elif not args.output:parser.error('export-pilot requires --output')
    else:export_pilot(args.output);print(args.output)