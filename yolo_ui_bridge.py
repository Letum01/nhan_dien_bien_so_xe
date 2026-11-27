"""Optional cached isolated YOLO; no Ultralytics import in main UI venv."""
from functools import lru_cache
from pathlib import Path
import json
import subprocess
import tempfile

import cv2

ROOT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/'experiments/yolo_round2_splitB_run01/weights/best.pt'
PYTHON=ROOT/'.venv-yolo/Scripts/python.exe'


def available():
    return PYTHON.is_file() and CHECKPOINT.is_file()


@lru_cache(maxsize=24)
def _infer(encoded,weights,mtime):
    with tempfile.TemporaryDirectory(prefix='alpr_yolo_') as folder:
        folder=Path(folder);source=folder/'image.png';target=folder/'result.json'
        source.write_bytes(encoded)
        process=subprocess.run([str(PYTHON),str(ROOT/'yolo_ui_worker.py'),str(source),str(target),weights],
                               cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        if process.returncode or not target.exists():raise RuntimeError((process.stderr+process.stdout)[-4000:])
        return json.loads(target.read_text(encoding='utf-8'))


def detect(image,weights):
    path=Path(weights).resolve()
    ok,encoded=cv2.imencode('.png',image)
    if not ok:raise ValueError('Could not encode image')
    result=_infer(encoded.tobytes(),str(path),path.stat().st_mtime_ns)
    return [(image[y:y+h,x:x+w].copy(),(x,y,w,h),row['confidence'])
            for row in result for x,y,w,h in [row['bbox']]]