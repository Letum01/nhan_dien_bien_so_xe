import sys

import numpy as np
import pytest

from dual_yolo_client import Worker


def test_timeout_restarts_and_discards_stale_responses(tmp_path):
    marker=tmp_path/'first'
    script=tmp_path/'worker.py'
    script.write_text('''import json,sys,time
from pathlib import Path
def send(**row): print(json.dumps(row),flush=True)
send(phase='boot');send(phase='ready',startup={},pid=__import__('os').getpid())
for line in sys.stdin:
    request=json.loads(line)
    marker=Path(sys.argv[1])
    if not marker.exists():
        marker.touch();time.sleep(10)
    send(id=request['id'], rows=[request['kind']])
''',encoding='utf-8')
    image=np.zeros((5,6,3),np.uint8)
    with Worker(timeout=2,command=[sys.executable,'-u',str(script),str(marker)]) as worker:
        with pytest.raises(TimeoutError):worker.request('plate',image)
        assert worker.process is None
        a=worker.request('plate',image)
        b=worker.request('character',image)
        assert a['cold'] and not b['cold'] and a['id']!=b['id']
        assert a['worker_info']['pid']==b['worker_info']['pid']
        assert a['rows']==['plate'] and b['rows']==['character']
    assert worker.process is None