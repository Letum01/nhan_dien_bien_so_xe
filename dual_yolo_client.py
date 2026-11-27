"""Serialized, restartable experimental worker. No image/result cache."""
import atexit
import json
from pathlib import Path
from queue import Empty,Queue
import subprocess
from tempfile import TemporaryDirectory
from threading import RLock,Thread
from time import perf_counter
from uuid import uuid4

import cv2

from yolo_ui_bridge import CHECKPOINT,PYTHON,ROOT

CHARACTER=ROOT/'experiments/yolo_character_round1/weights/best.pt'


class Worker:
    def __init__(self,timeout=90,command=None):
        self.timeout=timeout
        self.command=command or [str(PYTHON),'-u',str(ROOT/'dual_yolo_worker.py'),str(CHECKPOINT),str(CHARACTER)]
        self.lock=RLock(); self.process=None; self.info={}

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()

    def close(self):
        with self.lock:
            p,self.process=self.process,None
            self.info={}
            if p:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                p.stdin.close();p.stdout.close()

    def _receive(self):
        try:row=self.messages.get(timeout=self.timeout)
        except Empty as exc:raise TimeoutError('YOLO worker timed out') from exc
        if row is None:raise RuntimeError('YOLO worker exited; see worker log')
        return json.loads(row)

    def start(self):
        if self.process and self.process.poll() is None:return False
        self.close(); t=perf_counter(); messages=Queue()
        with (ROOT/'reports/dual_yolo_worker.log').open('ab') as log:
            p=subprocess.Popen(self.command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                               text=True,encoding='utf-8',bufsize=1)
        self.process=p;self.messages=messages
        def read():
            try:
                for line in p.stdout:messages.put(line)
            finally:messages.put(None)
        Thread(target=read,daemon=True).start()
        if self._receive().get('phase')!='boot':raise RuntimeError('Unexpected worker handshake')
        boot=perf_counter()-t
        self.info=self._receive()
        if self.info.get('phase')!='ready':raise RuntimeError('Worker did not reach ready')
        self.info['python_startup_seconds']=boot
        self.info['startup_seconds']=perf_counter()-t
        return True

    def request(self,kind,image):
        with self.lock,TemporaryDirectory(prefix='dual_yolo_') as tmp:
            path=Path(tmp)/'input.png'; started=perf_counter()
            if not cv2.imwrite(str(path),image):raise OSError('Could not encode input')
            try:
                cold=self.start(); token=uuid4().hex
                self.process.stdin.write(json.dumps({'id':token,'kind':kind,'path':str(path)})+'\n')
                self.process.stdin.flush(); row=self._receive()
                if row.get('id')!=token:raise RuntimeError('Mismatched YOLO response ID')
                if 'error' in row:raise RuntimeError(row['error'])
                return {**row,'cold':cold,'wall_seconds':perf_counter()-started,'worker_info':self.info.copy()}
            except Exception:
                self.close()
                raise


shared_worker=Worker()
atexit.register(shared_worker.close)