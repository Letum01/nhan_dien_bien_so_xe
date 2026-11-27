"""Fetch official Ultralytics v8.3.0 nano release for fine-tuning ONLY."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'


def main():
    from ultralytics import YOLO
    folder=ROOT/'pretrained';folder.mkdir(exist_ok=True)
    target=folder/'yolov8n.pt'
    if not target.exists():
        temporary=target.with_suffix('.download')
        urllib.request.urlretrieve(URL,temporary)
        model=YOLO(str(temporary.rename(folder/'yolov8n_download.pt')))
        # Save original downloaded bytes, not a reserialized/modified checkpoint.
        (folder/'yolov8n_download.pt').rename(target)
    model=YOLO(str(target))
    report={'source':URL,'checkpoint':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'classes':model.names,'parameters':sum(p.numel() for p in model.model.parameters()),
            'purpose':'generic pretrained initialization for fine-tuning ONLY; NOT a plate detector',
            'task':model.task}
    (ROOT/'reports/pretrained_nano.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='classes'},indent=2))


if __name__=='__main__':main()