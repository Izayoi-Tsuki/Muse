"""一次性脚本：抓取候选 MIDI 站点页面 HTML，供分析下载链接结构。"""
from pathlib import Path
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
OUT = Path(__file__).resolve().parents[1] / ".inspect"
OUT.mkdir(exist_ok=True)

PAGES = {
    "vgmusic_gc.html": "https://www.vgmusic.com/music/console/nintendo/gamecube/",
    "vgmusic_ds.html": "https://www.vgmusic.com/music/console/nintendo/ds/",
    "midishrine_ac.html": "https://www.midishrine.com/index.php?id=179",
    "bitmidi_search.html": "https://bitmidi.com/search?q=animal%20crossing",
}
for name, url in PAGES.items():
    try:
        r = requests.get(url, headers=UA, timeout=40)
        (OUT / name).write_text(r.text, encoding="utf-8")
        print(name, r.status_code, len(r.text))
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:100])