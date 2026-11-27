"""Explicit opt-in research artifacts, outside all training registries."""
from pathlib import Path
from uuid import uuid4
import json
import hashlib
from datetime import datetime,timezone
import cv2


def save_case(root, filename, payload, image, result, method, hashes, expected=''):
    target=Path(root)/uuid4().hex;target.mkdir(parents=True,exist_ok=False)
    (target/('original'+Path(filename).suffix.lower())).write_bytes(payload)
    if result['plate']:
        x,y,w,h=result['plate']['crop_box']
        if not cv2.imwrite(str(target/'crop.png'),image[y:y+h,x:x+w]):raise OSError('Cannot save crop')
    (target/'case.json').write_text(json.dumps({'scope':'opt-in development; not training, not blind test',
        'filename':Path(filename).name,'sha256':hashlib.sha256(payload).hexdigest(),'created_at':datetime.now(timezone.utc).isoformat(),
        'user_text_unreviewed':expected,'method':method,'hashes':hashes,'prediction':result},ensure_ascii=False,indent=2),encoding='utf-8')
    return target