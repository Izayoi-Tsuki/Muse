"""Lakh MIDI Dataset (lmd_full) 下载与解压，支持断点续传。

官方源限速严重（~20KB/s），改用 Hugging Face 镜像（与官方同文件，sha256 一致）。
lmd_full: 约 1.77GB，176,581 首 MIDI。
"""
from __future__ import annotations

import json
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lakh"
URL = "https://huggingface.co/datasets/AghaTizi/lakh_lmd_full/resolve/main/lmd_full.tar.gz"
ARCHIVE = RAW / "lmd_full.tar.gz"
EXTRACT_DIR = RAW / "lmd_full"
UA = {"User-Agent": "Mozilla/5.0"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    s.trust_env = False  # 绕过系统代理，官方源直连更快
    s.proxies = {"http": None, "https": None}
    return s


def download() -> None:
    """8 连接分段并行下载，每段独立断点续传。"""
    RAW.mkdir(parents=True, exist_ok=True)
    s = _session()
    head = s.head(URL, timeout=30, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    if not total:
        raise RuntimeError("no Content-Length")
    n_parts = 8
    sidecar = RAW / "lmd_full.parts.json"
    bounds = [(total * i // n_parts, total * (i + 1) // n_parts) for i in range(n_parts)]
    progress = json.loads(sidecar.read_text()) if sidecar.exists() else [0] * n_parts
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != total:
        with open(ARCHIVE, "wb") as f:
            f.truncate(total)
        progress = [0] * n_parts
    lock = threading.Lock()

    def fetch(i: int) -> None:
        start, end = bounds[i]
        part_len = end - start
        for attempt in range(30):
            if progress[i] >= part_len:
                return
            headers = dict(UA)
            headers["Range"] = f"bytes={start + progress[i]}-{end - 1}"
            try:
                with _session().get(URL, headers=headers, timeout=60, stream=True) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f"expected 206 for Range request, got {r.status_code}")
                    with open(ARCHIVE, "r+b") as f:
                        f.seek(start + progress[i])
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            with lock:
                                progress[i] += len(chunk)
                                done = sum(progress)
                            print(f"\r{done / 1e6:.1f}/{total / 1e6:.1f} MB ({done * 100 // total}%)", end="", flush=True)
                sidecar.write_text(json.dumps(progress))
            except Exception as e:
                print(f"\npart {i} attempt {attempt + 1} fail: {type(e).__name__}, resume at {progress[i]}")
        if progress[i] < part_len:
            raise RuntimeError(f"part {i} incomplete")

    with ThreadPoolExecutor(max_workers=n_parts) as ex:
        list(ex.map(fetch, range(n_parts)))
    sidecar.unlink(missing_ok=True)
    print(f"\ndownload complete: {ARCHIVE.stat().st_size} bytes")


def extract() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.mid")):
        print("already extracted, skip")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name.lower().endswith(".mid"):
                member.name = Path(member.name).name  # 扁平化，避免深层目录
                tf.extract(member, EXTRACT_DIR, filter="data")
                n += 1
                if n % 2000 == 0:
                    print(f"extracted {n} files", flush=True)
    print(f"extracted {n} midi files -> {EXTRACT_DIR}")


if __name__ == "__main__":
    for attempt in range(50):
        try:
            download()
            break
        except Exception as e:
            print(f"\nattempt {attempt + 1} failed: {type(e).__name__} {str(e)[:100]}, resuming...")
    if "--no-extract" not in sys.argv:
        extract()