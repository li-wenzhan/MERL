"""Bounded Ray startup waits and persistent tails of node-local runtime logs."""

import json
from pathlib import Path
import threading
import time


def startup_get(refs, *, stage, timeout=600, interval=30):
    import ray
    started = time.monotonic()
    print(f"[startup] {stage}: waiting (timeout={timeout}s)", flush=True)
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            resources = dict(cluster=ray.cluster_resources(), available=ray.available_resources())
            raise TimeoutError(f"Startup timed out at {stage} after {timeout}s; resources={resources}. "
                               "Inspect ray_logs in this experiment directory for worker/runtime_env errors.")
        try:
            result = ray.get(refs, timeout=min(interval, remaining))
            print(f"[startup] {stage}: ready after {time.monotonic() - started:.1f}s", flush=True)
            return result
        except ray.exceptions.GetTimeoutError:
            print(f"[startup] {stage}: still waiting after {time.monotonic() - started:.1f}s", flush=True)


class RayLogCapture:
    """Copy bounded text tails while a job runs, including before a hard ACP stop."""

    def __init__(self, ray_root, output, interval=30, tail_bytes=256 * 1024):
        self.ray_root = Path(ray_root)
        self.output = Path(output)
        self.interval, self.tail_bytes = interval, tail_bytes
        self.stop = threading.Event()
        self.thread = None
        self.seen = {}

    def snapshot(self):
        records = []
        for session in self.ray_root.glob("session_*"):
            if session.is_symlink() or not session.is_dir():
                continue
            for source in sorted((session / "logs").glob("*")):
                if source.is_symlink() or not source.is_file() or source.suffix not in (".log", ".out", ".err"):
                    continue
                stat = source.stat()
                relative = source.relative_to(self.ray_root)
                signature = (stat.st_size, stat.st_mtime_ns)
                if self.seen.get(str(relative)) != signature:
                    with source.open("rb") as handle:
                        handle.seek(max(0, stat.st_size - self.tail_bytes))
                        content = handle.read(self.tail_bytes)
                    target = self.output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    self.seen[str(relative)] = signature
                records.append(dict(path=str(relative), source_bytes=stat.st_size,
                                    retained_bytes=min(stat.st_size, self.tail_bytes)))
        if records:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "index.json").write_text(json.dumps(dict(captured_unix=time.time(), files=records), indent=2))

    def _run(self):
        while not self.stop.is_set():
            try:
                self.snapshot()
            except OSError as exc:
                print(f"[ray logs] capture warning: {exc}", flush=True)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=10)
        if not self.thread.is_alive():
            try:
                self.snapshot()
            except OSError as error:
                print(f"[ray logs] final capture warning: {error}", flush=True)
