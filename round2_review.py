"""Round-two queue validation/progress, without inferred human annotations."""
import csv
import hashlib
import json
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parent


def load_candidates(root=ROOT):
    root=Path(root)
    with (root/'reports/round2_plan/candidates.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    valid=[];errors=[];seen=set()
    for row in rows:
        path=(root/row['image_path']).resolve()
        reason=None
        if path in seen:reason='Duplicate candidate path'
        elif not path.is_file():reason='Missing image'
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:reason='Image hash changed'
        elif cv2.imread(str(path)) is None:reason='Image decode failed'
        seen.add(path)
        if reason:errors.append({'image_path':row['image_path'],'error':reason})
        else:valid.append({**row,'absolute_path':str(path)})
    return valid,errors


def progress(store,candidates):
    latest=store.all_latest();rows=[]
    for item in candidates:
        identity=item['sha256'];review=latest.get(identity);draft=store.draft(identity)
        if review and review.get('review_status')=='reviewed':status='reviewed'
        elif draft and draft[0].get('review_status') in ('needs_review','rejected'):status='needs_review'
        elif draft:status='draft'
        elif review:status='needs_review'
        else:status='not_started'
        rows.append({'image_path':item['image_path'],'status':status,
                     'source_group':(review or {}).get('source_group',''),
                     'split':(review or {}).get('split','UNASSIGNED')})
    return rows


def enforce_plan(candidate, latest, root=ROOT):
    """No rewriting frozen pilot images/group splits; new groups train/val only."""
    plan=json.loads((Path(root)/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    frozen=plan['frozen_pilot']
    if candidate['image_id'] in {r['image_id'] for r in frozen}:
        raise ValueError('Ảnh pilot đóng băng không được sửa trong chế độ vòng 2')
    group=candidate.get('source_group','').strip();split=candidate.get('split')
    if not group or group=='NEEDS_HUMAN_CONFIRMATION':raise ValueError('Cần source_group thực tế do người dùng xác minh')
    if split not in ('train','val'):raise ValueError('Ảnh bổ sung vòng 2 chỉ train/val; giữ nguyên test pilot')
    for row in frozen+list(latest.values()):
        if row.get('source_group')==group and row.get('split')!=split:
            raise ValueError('source_group xung đột split hiện có/đóng băng')
    # Duplicate leakage must be checked at the moment of approval too.
    from dataset_validation import check_split_leakage
    reviewed=[r for r in latest.values() if r.get('review_status')=='reviewed' and r['image_id']!=candidate['image_id']]
    check_split_leakage(reviewed+[candidate],Path(root))


def check_frozen_pilot(rows,root=ROOT):
    """Validate snapshot membership/labels without restoring or rewriting them."""
    plan=json.loads((Path(root)/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    actual={row['image_id']:row for row in rows}
    for frozen in plan['frozen_pilot']:
        current=actual.get(frozen['image_id'])
        if current is None:
            raise ValueError('Snapshot thiếu ảnh pilot đóng băng: '+frozen['image_path'])
        for field in ('boxes','source_group','split'):
            if current.get(field)!=frozen.get(field):
                raise ValueError(f'Pilot frozen {field} đã thay đổi: {frozen["image_path"]}; cần kiểm tra bằng người, không tự khôi phục')