"""Append-only registry for manually imported final-test images."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import warnings

from PIL import Image

from dataset_audit import fingerprint


def preview_bytes(data):
    if not 0 < len(data) <= 20*1024*1024:
        raise ValueError('Ảnh phải nhỏ hơn hoặc bằng 20 MiB')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            if image.format not in ('JPEG','PNG') or image.width*image.height>24_000_000:
                raise ValueError('Chỉ nhận JPEG/PNG tối đa 24 megapixel')
            image.load()
            return image.copy(), '.jpg' if image.format=='JPEG' else '.png'


def registered(db):
    db=Path(db).resolve()
    if not db.is_file():return []
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='final_intake'").fetchone():return []
        return [json.loads(row[0]) for row in con.execute('SELECT payload FROM final_intake ORDER BY rowid')]


def upload(db, filename, data, metadata, replaces=None):
    from final_ocr_dataset import ROOT, check_protocol, audit_candidates, conflicts, relative_image
    check_protocol()
    db=Path(db).resolve()
    _,extension=preview_bytes(data)
    assets=db.parent/(db.stem+'_assets'); assets.mkdir(exist_ok=True)
    import hashlib
    digest=hashlib.sha256(data).hexdigest(); path=assets/(digest+extension)
    with sqlite3.connect(db,timeout=30) as con:
        con.execute('CREATE TABLE IF NOT EXISTS final_intake (sha256 TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        con.execute('BEGIN IMMEDIATE')
        current=[json.loads(r[0]) for r in con.execute('SELECT payload FROM final_intake')]
        audit=audit_candidates(db,write=False)
        if replaces:
            target=next((r for r in audit['rows'] if relative_image(r['image'])==replaces),None)
            if not target or target.get('superseded_by') or any(r.get('replaces')==replaces for r in current):
                raise ValueError('Ứng viên thay thế không còn khả dụng')
            for (payload,) in con.execute('SELECT r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id) t ON r.image_id=t.image_id AND r.version=t.v'):
                row=json.loads(payload)
                if relative_image(row['image_path'])==replaces and row['review_status']=='reviewed':
                    raise ValueError('Không thay ảnh đã reviewed')
        if path.exists() or any(digest==r['sha256'] for r in audit['rows']+audit['known']):
            raise ValueError('Ảnh trùng SHA-256')
        # Exclusive create preserves an existing image; roll back new unregistered bytes on error.
        with path.open('xb') as file:file.write(data)
        try:
            _,phash,shape=fingerprint(path)
            if shape is None:raise ValueError('OpenCV không giải mã được ảnh')
            group=metadata.get('source_group','').strip()
            known_group_matches=[r for r in audit['known'] if group.casefold() and group.casefold()==r['source_group'].casefold()]
            if known_group_matches:raise ValueError('Source group trùng nguồn tham chiếu: '+known_group_matches[0]['image'])
            if conflicts(digest,phash,None,audit['known']+audit['rows']):
                raise ValueError('Ảnh gần trùng theo pHash (≤6 bit)')
            image=relative_image(path) or str(path)
            row={'image':image,'filename':Path(filename).name,'sha256':digest,'phash':f'{phash:016x}',
                 'status':'UNKNOWN','metadata':metadata,'replaces':replaces,'imported_at':datetime.now(timezone.utc).isoformat()}
            con.execute('INSERT INTO final_intake VALUES (?,?)',(digest,json.dumps(row,ensure_ascii=False)))
            con.commit()
        except Exception:
            path.unlink()
            raise
    return row