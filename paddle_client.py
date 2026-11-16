"""Serialized persistent Paddle worker with bounded waits and explicit recovery."""
import atexit
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid


class PaddleClient:
    """One model-owning child per client; timed-out children are reaped.

    Requests use JSON lines on stdin/stdout. Third-party library logs go to
    stderr, drained continuously to prevent pipe deadlocks. A failed request
    is reported, not silently retried; the next request starts a fresh worker.
    """

    def __init__(self, python: Path, script: Path, timeout: float = 120, cache_size: int = 32):
        self.python, self.script = Path(python), Path(script)
        self.timeout, self.cache_size = timeout, cache_size
        self._lock = threading.RLock()
        self._process = None
        self._messages = None
        self._logs = []
        self._cache = OrderedDict()
        atexit.register(self.close)

    @staticmethod
    def _read_messages(pipe, messages):
        try:
            for line in pipe:
                try:
                    messages.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            messages.put({"event": "eof"})

    def _drain_logs(self, pipe):
        for line in pipe:
            self._logs.append(line.rstrip())
            del self._logs[:-80]

    def _receive(self):
        try:
            message = self._messages.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Paddle worker exceeded {self.timeout}s; " + "\n".join(self._logs[-8:])) from exc
        if message.get("event") == "eof":
            raise RuntimeError("Paddle worker exited: " + "\n".join(self._logs[-12:]))
        return message

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return 0.0
        self.close()
        self._logs = []
        self._messages = queue.Queue()
        start = time.perf_counter()
        self._process = subprocess.Popen(
            [str(self.python), "-u", str(self.script), "--serve"],
            cwd=str(self.script.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._read_messages, args=(self._process.stdout, self._messages), daemon=True).start()
        threading.Thread(target=self._drain_logs, args=(self._process.stderr,), daemon=True).start()
        ready = self._receive()
        if ready.get("event") != "ready":
            raise RuntimeError(f"Paddle initialization failed: {ready}")
        return time.perf_counter() - start

    def request(self, encoded: bytes, config: str = "mobile-en-v1", use_cache: bool = True):
        """Return fresh execution timings even on a cache hit, never stale latency."""
        with self._lock:
            start = time.perf_counter()
            # Include interpreter/script content/config in cache identity.
            key = hashlib.sha256(str(self.python).encode() + self.script.read_bytes()
                                 + config.encode() + encoded).hexdigest()
            if use_cache and key in self._cache:
                result = copy.deepcopy(self._cache[key])
                self._cache.move_to_end(key)
                result["timings"] = {"cache_hit": True, "cold_start_seconds": 0.0,
                                     "inference_seconds": 0.0, "decode_seconds": 0.0,
                                     "worker_pid": None, "request_seconds": time.perf_counter() - start}
                return result
            try:
                cold_seconds = self._start()
                import base64
                request_id = uuid.uuid4().hex
                message = {"id": request_id, "png_base64": base64.b64encode(encoded).decode("ascii")}
                self._process.stdin.write(json.dumps(message) + "\n")
                self._process.stdin.flush()
                response = self._receive()
                if response.get("id") != request_id:
                    raise RuntimeError("Paddle protocol request ID mismatch")
                if response.get("error"):
                    raise RuntimeError(response["error"])
                result = response["result"]
                result["timings"].update({"cache_hit": False, "cold_start_seconds": cold_seconds,
                                           "request_seconds": time.perf_counter() - start})
                if use_cache:
                    self._cache[key] = copy.deepcopy(result)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
                return result
            except Exception:
                self.close()
                raise

    def close(self):
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe:
                        pipe.close()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()