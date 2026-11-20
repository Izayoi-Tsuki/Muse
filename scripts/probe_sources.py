"""一次性探测脚本：检查各 MIDI 数据源在本机网络下的可达性。"""
import os
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGETS = [
    ("vgmusic_gc", "https://www.vgmusic.com/music/console/nintendo/gamecube/", "GET"),
    ("ninsheetmusic", "https://www.ninsheetmusic.org/browse/series/AnimalCrossing", "GET"),
    ("lmd_matched", "https://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz", "HEAD"),
    ("lmd_full", "https://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz", "HEAD"),
    ("pop909_github", "https://github.com/music-x-lab/POP909-Dataset", "GET"),
    ("onlinesequencer", "https://onlinesequencer.net/2166746", "GET"),
]

print("proxy env:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})
for name, url, method in TARGETS:
    for trust_env in (True, False):
        tag = "proxy" if trust_env else "direct"
        try:
            s = requests.Session()
            s.trust_env = trust_env
            r = s.request(method, url, headers=UA, timeout=25, allow_redirects=True, stream=True)
            size = r.headers.get("Content-Length", "?")
            print(f"{name:16s} [{tag:6s}] {r.status_code} len={size}")
            r.close()
            break
        except Exception as e:
            print(f"{name:16s} [{tag:6s}] FAIL {type(e).__name__}: {str(e)[:90]}")