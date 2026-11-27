"""Run using .venv-yolo; actual GPU matmul + locally constructed nano forward."""
import json
from pathlib import Path
import time

import torch
import ultralytics


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    a=torch.randn(256,256,device='cuda');b=a@a.T;torch.cuda.synchronize()
    assert torch.isfinite(b).all().item()
    # Architecture only: no weights downloaded, no detector quality assertion.
    architecture=Path(ultralytics.__file__).parent/'cfg/models/v8/yolov8.yaml'
    from ultralytics import YOLO
    model=YOLO(str(architecture)).model.to('cuda').eval()
    start=time.perf_counter()
    with torch.no_grad():result=model(torch.zeros(1,3,640,640,device='cuda'))
    torch.cuda.synchronize()
    report={'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,
            'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),
            'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
            'cuda_matmul':'PASS','architecture_forward':'PASS','forward_seconds':time.perf_counter()-start,
            'trained_weights_used':False,'training':'BLOCKED no reviewed labels'}
    root=Path(__file__).resolve().parent
    (root/'reports/yolo_cuda.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()