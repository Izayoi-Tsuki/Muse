"""动森 MIDI 收集器：BitMidi 搜索下载 + vgmusic 分区下载。

BitMidi 搜索页内嵌 window.initStore JSON，含每首曲目的 downloadUrl(/uploads/{id}.mid)。
vgmusic 索引页为 HTML 表格，按游戏分区列出 .mid 直链。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
ROOT = Path(__file__).resolve().parents[1]
AC_DIR = ROOT / "data" / "raw" / "animal_crossing"

BITMIDI_QUERIES = [
    "animal crossing",
    "animal crossing new horizons",
    "animal crossing new leaf",
    "animal crossing wild world",
    "animal crossing city folk",
    "k.k. slider",
]
VGMUSIC_PAGES = {
    "gc": "http://www.vgmusic.com/music/console/nintendo/gamecube/",
    "ds": "http://www.vgmusic.com/music/console/nintendo/ds/",
}
# vgmusic 表格中的游戏分区标题（行首单元格文本）
VGMUSIC_SECTIONS = {"gc": ["Animal Crossing"], "ds": ["Animal Crossing: Wild World"]}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    # 绕过系统/注册表代理，部分站点经代理会超时
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()[:120]


def _download(s: requests.Session, url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for i in range(retries):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 200 and r.content[:4] in (b"MThd", b"RIFF"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                return True
            print(f"  skip {url} -> HTTP {r.status_code}, head={r.content[:4]!r}")
            return False
        except Exception as e:
            print(f"  retry {i + 1}/{retries} {url}: {type(e).__name__}")
            time.sleep(2 * (i + 1))
    return False


def collect_bitmidi(s: requests.Session) -> int:
    out = AC_DIR / "bitmidi"
    seen: dict[int, str] = {}
    for q in BITMIDI_QUERIES:
        page = 0
        while True:
            url = f"https://bitmidi.com/search?q={requests.utils.quote(q)}&page={page}"
            try:
                r = s.get(url, timeout=40)
                m = re.search(r"window\.initStore\s*=\s*(\{.*?\})\s*</script>", r.text, re.S)
                if not m:
                    print(f"[bitmidi] no initStore for '{q}' page {page}")
                    break
                store = json.loads(m.group(1))
            except Exception as e:
                print(f"[bitmidi] fail '{q}' page {page}: {type(e).__name__} {str(e)[:80]}")
                break
            midis = store.get("data", {}).get("midis", {})
            if not midis:
                break
            for slug, info in midis.items():
                seen[info["id"]] = info["name"]
            page += 1
            if page >= 20:  # 安全上限
                break
            time.sleep(0.5)
        print(f"[bitmidi] query '{q}' done, cumulative unique: {len(seen)}")
    ok = 0
    for mid, name in sorted(seen.items()):
        dest = out / f"{mid:06d}_{_safe_name(name)}"
        if not dest.suffix.lower() == ".mid":
            dest = dest.with_suffix(".mid")
        if _download(s, f"https://bitmidi.com/uploads/{mid}.mid", dest):
            ok += 1
        time.sleep(0.3)
    print(f"[bitmidi] downloaded {ok}/{len(seen)}")
    return ok


def collect_vgmusic(s: requests.Session) -> int:
    ok = 0
    for console, index_url in VGMUSIC_PAGES.items():
        try:
            r = s.get(index_url, timeout=90)
            html = r.text
        except Exception as e:
            print(f"[vgmusic:{console}] index fail: {type(e).__name__} {str(e)[:80]}")
            continue
        # 逐行扫描表格：分区标题行形如 <td ...>Animal Crossing</td>，
        # 曲目行含 <a href="xxx.mid">
        in_section = False
        links: list[tuple[str, str]] = []
        for line in html.splitlines():
            title_m = re.search(r"<td[^>]*>\s*([^<>]+?)\s*</td>\s*$", line)
            if title_m and not re.search(r'href=', line):
                text = title_m.group(1).strip()
                if text in VGMUSIC_SECTIONS[console]:
                    in_section = True
                    continue
                # 遇到下一个非空标题即离开当前分区
                if text and in_section and not re.match(r"^\d", text):
                    in_section = False
            if in_section:
                for href, title in re.findall(r'<a href="([^"]+\.mid)"[^>]*>([^<]*)</a>', line):
                    links.append((urljoin(index_url, href), title.strip() or Path(href).stem))
        print(f"[vgmusic:{console}] found {len(links)} files in section")
        for url, title in links:
            dest = AC_DIR / f"vgmusic_{console}" / f"{_safe_name(title)}.mid"
            if _download(s, url, dest):
                ok += 1
            time.sleep(0.3)
    return ok


def main() -> None:
    AC_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    n1 = collect_bitmidi(s)
    n2 = collect_vgmusic(s)
    print(f"TOTAL animal_crossing: bitmidi={n1}, vgmusic={n2}")


if __name__ == "__main__":
    main()