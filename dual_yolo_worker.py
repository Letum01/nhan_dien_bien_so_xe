"""Private line-delimited IPC worker in the PyTorch environment."""
from time import perf_counter
started=perf_counter()
import contextlib
import json
import sys


def main():
    wire=sys.stdout
    def send(row):
        wire.write(json.dumps(row)+'\n'); wire.flush()
    with contextlib.redirect_stdout(sys.stderr):
        send({'phase':'boot'})
        t=perf_counter()
        import cv2
        import numpy as np
        import torch
        from yolo_detector import load_plate_yolo,detect_yolo
        from yolo_character_worker import load_model,predict
        from ultralytics import YOLO  # measure import separately from checkpoint loads
        startup={'imports':perf_counter()-t}
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        plate,character=sys.argv[1:3]
        t=perf_counter(); load_plate_yolo(plate); startup['plate_load']=perf_counter()-t
        t=perf_counter(); char=load_model(character); startup['character_load']=perf_counter()-t
        t=perf_counter(); blank=np.zeros((640,640,3),np.uint8)
        detect_yolo(blank,plate); predict(blank,char); torch.cuda.synchronize(0)
        startup['warmup']=perf_counter()-t
        send({'phase':'ready','gpu':torch.cuda.get_device_name(0),'pid':__import__('os').getpid(),'startup':startup})
        for line in sys.stdin:
            request=json.loads(line)
            image=cv2.imread(request['path'])
            if image is None:
                send({'id':request['id'],'error':'Unreadable image'}); continue
            torch.cuda.synchronize(0); t=perf_counter()
            if request['kind']=='plate':
                rows=[{'bbox':b,'score':s} for _,b,s in detect_yolo(image,plate)]
            elif request['kind']=='character':
                rows=predict(image,char)
            else:
                send({'id':request['id'],'error':'Unknown request'}); continue
            torch.cuda.synchronize(0)
            send({'id':request['id'],'rows':rows,'worker_seconds':perf_counter()-t,
                  'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0),
                  'peak_allocated':torch.cuda.max_memory_allocated(0),'peak_reserved':torch.cuda.max_memory_reserved(0)})


if __name__=='__main__':main()