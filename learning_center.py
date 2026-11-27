"""Local feedback store; research records never feed a training loader."""
from collections import Counter
from contextlib import closing, contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import cv2
import numpy as np

from final_ocr_intake import preview_bytes

CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SPLITS = ('unassigned', 'train', 'development', 'test')
MODELS = ('cnn', 'svm', 'rf')


def decode_upload(data):
    preview_bytes(data)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh.')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tile = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff = cv2.dct(tile)[:8, :8].ravel()
    phash = sum(int(bit) << i for i, bit in enumerate(coeff > np.median(coeff[1:])))
    return image, hashlib.sha256(data).hexdigest(), f'{phash:016x}'


def normalized(value):
    text = re.sub(r'[\s.-]', '', value.upper())
    if not re.fullmatch('[0-9A-Z]{1,16}', text):
        raise ValueError('Nhập 1–16 ký tự A–Z hoặc 0–9; có thể dùng dấu cách, chấm, gạch.')
    return text


def alignment(text, raw):
    boxes = raw.get('glyphs', [])
    if (raw.get('plate_count') != 1 or raw.get('plate_score', 0) < .60 or
            not boxes or len(boxes) != len(text) or raw.get('geometry_flags') or raw.get('poor_crop')):
        return 'ALIGNMENT_REJECTED', [], 'Số box, xác suất detector hoặc hình học chưa đạt.'
    for g, char in zip(boxes, text, strict=True):
        ranks = g.get('top_k', {})
        if (any(not ranks.get(m) for m in MODELS) or g['score'] < .50 or not .03 <= g['foreground'] <= .75 or
                sum(char in ranks[m][:3] for m in MODELS) < 2 or
                not any(char == ranks[m][0] for m in MODELS)):
            return 'ALIGNMENT_REJECTED', [], 'Thiếu bằng chứng pixel hoặc top-k tại một vị trí.'
    return 'AUTO_ALIGNED', [{'slot': i, 'bbox_crop_xywh': g['bbox'], 'class': c} for i, (g,c) in enumerate(zip(boxes,text))], 'Đề xuất tự động, chưa kiểm định box.'


def priority(raw):
    # Ranks uncertainty within a backend, never compares raw scores across models.
    return sum(bool(raw.get(k)) for k in ('disagreement','low_cnn_margin','rare_candidates',
                                         'geometry_flags','low_detector','poor_crop')) + int(not raw.get('plate_count'))


class Feedback:
    def __init__(self, path, references=()):
        self.path = Path(path); self.refs = tuple(references)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, phash TEXT NOT NULL, png BLOB NOT NULL,
              metadata TEXT NOT NULL, raw TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions (id TEXT REFERENCES samples(id) ON DELETE CASCADE,
              revision INTEGER NOT NULL, text TEXT, status TEXT NOT NULL, alignment TEXT NOT NULL,
              reason TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY(id, revision));''')

    @contextmanager
    def db(self, write=False):
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON'); db.execute('PRAGMA secure_delete=ON')
            with db:
                if write: db.execute('BEGIN IMMEDIATE')
                yield db

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def check(self, digest, phash, group, split, current=()):
        for row in (*self.refs, *current):
            if row['sha256'] == digest:
                raise ValueError('Ảnh trùng SHA-256.')
            if row.get('phash') and (int(phash,16)^int(row['phash'],16)).bit_count() <= 6:
                raise ValueError('Ảnh gần trùng pHash (≤6 bit); không nhập tự động.')
            if group and row.get('group','').strip().casefold() == group:
                if row.get('protected') or row.get('split') != split:
                    raise ValueError('Source group xung đột với tập tham chiếu hoặc split khác.')

    def add(self, data, *, metadata, consent):
        if consent is not True:
            raise ValueError('Cần chủ động đồng ý lưu ảnh nghiên cứu.')
        im,digest,phash = decode_upload(data)
        md = {k: str(metadata.get(k) or '').strip() for k in ('source','group','capture_date','rights','split')}
        md['group'] = md['group'].casefold(); md['consent'] = 'OPT_IN'
        if any(len(v)>240 for v in md.values()) or md['split'] not in SPLITS or md['rights'] not in ('unknown','research_only','research_and_training'):
            raise ValueError('Metadata không hợp lệ.')
        if md['capture_date']: date.fromisoformat(md['capture_date'])
        ok,png=cv2.imencode('.png',im)
        if not ok: raise ValueError('Không mã hóa được ảnh.')
        md['stored_png_sha256']=hashlib.sha256(png).hexdigest()
        status='UNREVIEWED' if md['source'] and md['group'] and md['rights']!='unknown' else 'PROVENANCE_BLOCKED'
        md['provenance'] = 'SELF_REPORTED' if status=='UNREVIEWED' else 'PROVENANCE_BLOCKED'
        with self.db(write=True) as db:
            current=[dict(sha256=r['id'],phash=r['phash'],**json.loads(r['metadata'])) for r in db.execute('SELECT id,phash,metadata FROM samples')]
            self.check(digest,phash,md['group'],md['split'],current)
            db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?)',(digest,phash,png.tobytes(),json.dumps(md), '{}',self.now()))
            db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?,?)',(digest,1,None,status,'[]','',self.now()))
        return digest

    def set_prediction(self, key, raw):
        with self.db(write=True) as db:
            row=db.execute('SELECT metadata,raw FROM samples WHERE id=?',(key,)).fetchone()
            if row is None or json.loads(row['metadata'])['split']=='test' or row['raw']!='{}':
                raise ValueError('Không được chạy lại hoặc lưu dự đoán cho ảnh test.')
            db.execute('UPDATE samples SET raw=? WHERE id=?',(json.dumps(raw,allow_nan=False),key))

    def confirm(self, key, text, *, confirmed=False, skip=False):
        if not skip and confirmed is not True: raise ValueError('Cần xác nhận text đọc từ ảnh.')
        text=None if skip else normalized(text)
        with self.db(write=True) as db:
            row=db.execute('SELECT metadata,raw FROM samples WHERE id=?',(key,)).fetchone()
            if row is None: raise ValueError('Mẫu không tồn tại.')
            revision=db.execute('SELECT MAX(revision) FROM revisions WHERE id=?',(key,)).fetchone()[0]+1
            raw=json.loads(row['raw']); md=json.loads(row['metadata']); boxes=[]; reason=''
            if skip: status='AMBIGUOUS'; reason='Người dùng bỏ qua.'
            elif raw and md['split']!='test': status,boxes,reason=alignment(text,raw)
            else: status='TEXT_CONFIRMED'
            db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?,?)',(key,revision,text,status,json.dumps(boxes),reason,self.now()))
        return status

    def samples(self):
        with self.db() as db:
            return [self.unpack(r) for r in db.execute('''SELECT s.id,s.phash,s.metadata,s.raw,r.revision,r.text,r.status,r.alignment,r.reason
                FROM samples s JOIN revisions r ON s.id=r.id WHERE r.revision=(SELECT MAX(b.revision) FROM revisions b WHERE b.id=s.id)''')]

    @staticmethod
    def unpack(row):
        return {k:json.loads(v) if k in ('metadata','raw','alignment') else v for k,v in dict(row).items()}

    def pixels(self, key):
        with self.db() as db:
            row=db.execute('SELECT png FROM samples WHERE id=?',(key,)).fetchone()
            if row is None: raise ValueError('Mẫu không tồn tại.')
            return row['png']

    def delete(self,key,*,confirmed=False):
        if confirmed is not True: raise ValueError('Cần xác nhận xóa.')
        with self.db(write=True) as db: db.execute('DELETE FROM samples WHERE id=?',(key,))

    def export(self,key,*,allowed=False):
        if allowed is not True: raise ValueError('Cần xác nhận xuất.')
        with self.db() as db:
            row=db.execute('SELECT id,phash,metadata,raw,created FROM samples WHERE id=?',(key,)).fetchone()
            if row is None: raise ValueError('Mẫu không tồn tại.')
            result=self.unpack(row)
            result['revisions']=[self.unpack(r) for r in db.execute('SELECT * FROM revisions WHERE id=? ORDER BY revision',(key,))]
        return {'purpose':'RESEARCH_ONLY', 'training_allowed':False, 'sample':result}

    def training_snapshot(self):
        raise ValueError('BLOCKED: chưa có benchmark mới đóng băng, kiểm định box alignment độc lập và phê duyệt snapshot.')


def dashboard(rows):
    return {'images':len(rows),'source_documented':sum(bool(r['metadata']['source'] and r['metadata']['group'] and r['metadata']['rights']!='unknown') for r in rows),
            'confirmed_text':sum(r['text'] is not None for r in rows),'groups_recorded':len({r['metadata']['group'] for r in rows if r['metadata']['group']}),
            'states':dict(Counter(r['status'] for r in rows)), 'splits':dict(Counter(r['metadata']['split'] for r in rows)),
            'proposed_glyphs':sum(len(r['alignment']) for r in rows),'confirmed_text_classes':dict(Counter(''.join(r['text'] or '' for r in rows))),
            'layout_proxy':dict(Counter(r['raw'].get('layout','unknown') for r in rows))}