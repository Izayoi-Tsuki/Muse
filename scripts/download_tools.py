"""下载渲染工具：FluidSynth Windows 二进制 + GeneralUser GS SoundFont。

GitHub API 匿名限流，改用固定 release 直链 + git clone。
"""
from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
UA = {"User-Agent": "Mozilla/5.0"}
FS_URL = "https://github.com/fluidsynth/fluidsynth/releases/download/v2.6.2/fluidsynth-v2.6.2-win10-x64-cpp11.zip"
GS_REPO = "https://github.com/mrbumpy409/GeneralUser-GS.git"


def _get(url: str, **kw) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s.get(url, headers=UA, timeout=120, **kw)


def main() -> None:
    fs_dir = TOOLS / "fluidsynth"
    if not (fs_dir / "bin" / "fluidsynth.exe").exists():
        fetch_zip(FS_URL, fs_dir)
        # 压缩包带一层顶级目录，规整到 bin/ 结构
        if not (fs_dir / "bin" / "fluidsynth.exe").exists():
            for sub in fs_dir.iterdir():
                if sub.is_dir() and (sub / "bin" / "fluidsynth.exe").exists():
                    for item in sub.iterdir():
                        item.rename(fs_dir / item.name)
                    break
    sf_dir = TOOLS / "soundfonts"
    has_sf = sf_dir.exists() and any(sf_dir.rglob("*.sf2"))
    if not has_sf:
        subprocess.run(["git", "clone", "--depth", "1", GS_REPO, str(sf_dir / "GeneralUser-GS")], check=True)
    print("tools ready:", (fs_dir / "bin" / "fluidsynth.exe").exists(),
          [str(p) for p in sf_dir.rglob("*.sf2")])


def fetch_zip(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    r = _get(url)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest)
    print(f"extracted -> {dest}")


if __name__ == "__main__":
    main()