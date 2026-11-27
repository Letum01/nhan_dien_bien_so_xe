import json
from pathlib import Path
import sys

import pytest

from paddle_client import PaddleClient


@pytest.fixture
def worker_script(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import sys,json,os,time,base64
print(json.dumps({"event":"ready"}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 value=base64.b64decode(r["png_base64"])
 if value==b"crash": sys.exit(3)
 if value==b"slow": time.sleep(2)
 print(json.dumps({"id":r["id"],"result":{"raw_text":"test","timings":{"worker_pid":os.getpid(),"inference_seconds":0.01,"decode_seconds":0}}}),flush=True)
''', encoding="utf-8")
    return script


def test_reuses_pid_caches_and_invalidates_by_config(worker_script):
    client = PaddleClient(Path(sys.executable), worker_script, timeout=5)
    try:
        first = client.request(b"one")
        second = client.request(b"two")
        assert first["timings"]["worker_pid"] == second["timings"]["worker_pid"]
        assert first["timings"]["cold_start_seconds"] > 0
        assert second["timings"]["cold_start_seconds"] == 0
        cached = client.request(b"one")
        assert cached["timings"]["cache_hit"]
        assert cached["timings"]["inference_seconds"] == 0
        assert not client.request(b"one", config="v2")["timings"]["cache_hit"]
    finally:
        client.close()


def test_dead_worker_restarts_on_next_request(worker_script):
    client = PaddleClient(Path(sys.executable), worker_script, timeout=5)
    try:
        old_pid = client.request(b"one")["timings"]["worker_pid"]
        with pytest.raises(RuntimeError, match="exited"):
            client.request(b"crash")
        assert client.request(b"new")["timings"]["worker_pid"] != old_pid
    finally:
        client.close()


def test_timeout_kills_and_next_request_recovers(worker_script):
    client = PaddleClient(Path(sys.executable), worker_script, timeout=5)
    try:
        client.request(b"one")
        client.timeout = .15
        with pytest.raises(TimeoutError):
            client.request(b"slow")
        assert client._process is None
        client.timeout = 5
        assert client.request(b"after")["raw_text"] == "test"
    finally:
        client.close()