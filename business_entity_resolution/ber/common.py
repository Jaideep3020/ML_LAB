from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import duckdb
import psutil


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sqlpath(path):
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


class Context:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.cfg = read_json(config) if not isinstance(config, dict) else config
        self.root.mkdir(parents=True, exist_ok=True)
        for d in ("data", "cache", "reports", "models", "output", "temp"):
            (self.root / d).mkdir(exist_ok=True)
        self.config_hash = fingerprint(self.cfg)

    def db(self):
        c = duckdb.connect(str(self.root / "cache" / "data.duckdb"))
        c.execute("SET memory_limit=?", [self.cfg["memory_limit"]])
        c.execute("SET threads=?", [self.cfg["threads"]])
        c.execute("SET temp_directory=?", [str(self.root / "temp")])
        c.execute("SET preserve_insertion_order=false")
        return c

    def guard_disk(self, additional=0):
        free = shutil.disk_usage(self.root).free
        minimum = self.cfg["disk_reserve_gb"] * (1 << 30) + additional
        if free < minimum:
            raise RuntimeError(f"Insufficient disk: {free / 2**30:.1f} GiB free; {minimum / 2**30:.1f} GiB required")

    def event(self, stage, **kw):
        memory = psutil.Process().memory_info()
        event = dict(stage=stage, time=time.time(), rss_gb=memory.rss / 2**30,
                     peak_rss_gb=getattr(memory,'peak_wset',memory.rss)/2**30,
                     config_hash=self.config_hash, **kw)
        with open(self.root / "reports" / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)


def batches(reader, size=10000):
    while rows := reader.fetchmany(size):
        yield rows
