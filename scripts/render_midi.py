"""MIDI -> WAV 渲染：pyfluidsynth 加载本地 DLL + SoundFont，file 音频驱动离线渲染。

（系统策略禁止从工作区执行 exe，故不走 fluidsynth CLI，直接 ctypes 加载 DLL。）
工具由 scripts/download_tools.py 下载到 tools/。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS_BIN = ROOT / "tools" / "fluidsynth" / "bin"
SF_DIR = ROOT / "tools" / "soundfonts"

# 必须在 import fluidsynth 前把 DLL 目录加入搜索路径
os.environ["PATH"] = str(FS_BIN) + os.pathsep + os.environ.get("PATH", "")


def find_soundfont() -> Path | None:
    if not SF_DIR.exists():
        return None
    for p in sorted(SF_DIR.rglob("*.sf2")):
        return p
    return None


def render(midi_path: Path, wav_path: Path | None = None, gain: float = 0.8) -> Path | None:
    sf = find_soundfont()
    if not (FS_BIN / "libfluidsynth-3.dll").exists() or sf is None:
        print("fluidsynth/soundfont not ready, run scripts/download_tools.py first")
        return None
    import time

    import fluidsynth

    wav_path = wav_path or midi_path.with_suffix(".wav")
    fs = fluidsynth.Synth(gain=gain, samplerate=44100.0)
    try:
        fs.setting("audio.driver", "file")
        fs.setting("audio.file.name", str(wav_path))
        fs.setting("audio.file.format", "s16")
        fs.setting("audio.file.type", "wav")
        fs.setting("midi.driver", "none")  # 离线渲染无需物理 MIDI 输入
        sfid = fs.sfload(str(sf))
        if sfid < 0:
            print(f"soundfont load failed: {sf}")
            return None
        fs.start()  # 创建 file 音频驱动（离线渲染线程）
        if fs.play_midi_file(str(midi_path)) == fluidsynth.FLUID_FAILED:
            print(f"midi load failed: {midi_path}")
            return None
        # play_midi_file 非阻塞：轮询播放器状态直到完成
        for _ in range(36000):  # 最长 1 小时
            if fluidsynth.fluid_player_get_status(fs.player) == fluidsynth.FLUID_PLAYER_DONE:
                break
            time.sleep(0.1)
        time.sleep(0.5)  # 等 file 驱动 flush 尾音
    finally:
        fs.delete()
    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        print("render produced no audio")
        return None
    return wav_path


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        out = render(Path(arg))
        print(out or f"FAILED: {arg}")