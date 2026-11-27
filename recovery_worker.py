"""Experimental CUDA service. Frozen V1 worker is deliberately untouched."""
import contextlib
import json
import math
import sys
from time import perf_counter


def main():
    wire=sys.stdout
    def send(message):
        print(json.dumps(message),file=wire,flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        send({'phase':'boot'})
        import cv2
        import numpy as np
        import torch
        from yolo_detector import load_plate_yolo
        from yolo_character_worker import load_model
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        models={'plate':load_plate_yolo(sys.argv[1]),'character':load_model(sys.argv[2])}

        @torch.inference_mode()
        def predict(im,key,size):
            plate=key=='plate'; h,w=im.shape[:2]
            result=models[key].predict(im,imgsz=size,device=0,half=False,conf=.4 if plate else .25,
                                       iou=.7,max_det=300 if plate else 40,verbose=False)[0]
            rows=[]
            for (a,b,c,d),score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                x,y=max(0,int(a) if plate else math.floor(a)),max(0,int(b) if plate else math.floor(b))
                r,s=min(w,int(c) if plate else math.ceil(c)),min(h,int(d) if plate else math.ceil(d))
                if r>x and s>y:rows.append({'bbox':[x,y,r-x,s-y],'score':score})
            return rows

        for key in models:predict(np.zeros((640,640,3),np.uint8),key,640)
        send({'phase':'ready','gpu':torch.cuda.get_device_name(0),'startup':{}})
        allowed={'plate':(640,960,1280),'character':(640,800,960)}
        for line in sys.stdin:
            req=json.loads(line); key, _, size=req['kind'].partition(':'); size=int(size)
            im=cv2.imread(req['path'])
            if im is None or size not in allowed.get(key,()):
                send({'id':req['id'],'error':'Invalid image or scale'});continue
            torch.cuda.synchronize();start=perf_counter();rows=predict(im,key,size);torch.cuda.synchronize()
            send({'id':req['id'],'rows':rows,'worker_seconds':perf_counter()-start,
                  'peak_reserved':torch.cuda.max_memory_reserved()})


if __name__=='__main__':main()