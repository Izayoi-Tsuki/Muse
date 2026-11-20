"""MIDI 清洗与规范化：过滤、1/16 拍量化、小编制乐器映射、调性归一。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import symusic

# Krumhansl-Kessler 调性轮廓（C 起点的 12 个音级权重）
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# GM program 区间 -> ensemble 名
PROGRAM_RANGES = [
    (0, 7, "piano"), (8, 15, "mallet"), (16, 23, "piano"),      # 钢琴 / 色彩打击 / 风琴
    (24, 31, "guitar"), (32, 39, "bass"),                        # 吉他 / 贝斯
    (40, 55, "strings"),                                         # 弦乐 / 合唱
    (56, 79, "strings"),                                         # 铜管 / 簧管 / 吹管 -> 弦乐垫底
    (80, 103, "piano"),                                          # synth lead/pad -> 钢琴
    (104, 111, "mallet"),                                        # 民族乐器 -> 马林巴
    (112, 119, "mallet"),                                        # 打击音效 -> 马林巴
    (120, 127, "piano"),                                         # 音效 -> 钢琴
]


@dataclass
class SongMeta:
    song_id: str
    source: str
    path: str
    n_notes: int = 0
    duration_sec: float = 0.0
    tempo: float = 0.0
    time_signature: tuple = (4, 4)
    orig_key: str = "?"
    key_shift: int = 0
    programs: list = field(default_factory=list)
    n_tracks: int = 0
    token_len: int = 0


def load_score(path: Path) -> symusic.Score | None:
    try:
        score = symusic.Score.from_file(str(path))
        if score.note_num() == 0:
            return None
        # 记谱伪影修正：POP909 等数据集把 4/4 记成 1/4、2/4、2/2，
        # 统一重写为 4/4（仅改拍号标记，不改音符时序）
        for ts in score.time_signatures:
            if (ts.numerator, ts.denominator) in ((1, 4), (2, 4), (2, 2)):
                ts.numerator, ts.denominator = 4, 4
        return score
    except Exception:
        return None


def duration_seconds(score: symusic.Score) -> float:
    """按 tempo map 积分计算实际时长（秒）。symusic 的 Tempo.tempo 单位为 BPM。"""
    tpq = score.tpq
    end_tick = score.end()
    tempos = sorted(score.tempos, key=lambda t: t.time)
    if not tempos:
        tempos = [symusic.Tempo(0, 120.0)]  # 默认 120 BPM
    sec, prev_tick, prev_bpm = 0.0, 0, tempos[0].tempo
    for t in tempos[1:]:
        if t.time >= end_tick:
            break
        sec += (t.time - prev_tick) / tpq * 60.0 / prev_bpm
        prev_tick, prev_bpm = t.time, t.tempo
    sec += (end_tick - prev_tick) / tpq * 60.0 / prev_bpm
    return sec


def dominant_tempo(score: symusic.Score) -> float:
    """按时长加权的主体 tempo（BPM）。"""
    tpq = score.tpq
    end_tick = max(score.end(), 1)
    tempos = sorted(score.tempos, key=lambda t: t.time) or [symusic.Tempo(0, 120.0)]
    weights, values = [], []
    for i, t in enumerate(tempos):
        nxt = tempos[i + 1].time if i + 1 < len(tempos) else end_tick
        weights.append(max(0, min(nxt, end_tick) - t.time))
        values.append(t.tempo)
    w = np.array(weights, dtype=np.float64)
    if w.sum() == 0:
        return values[0]
    return float(np.average(values, weights=w))


def time_signature(score: symusic.Score) -> tuple[int, int]:
    if len(score.time_signatures):
        ts = score.time_signatures[0]
        return (ts.numerator, ts.denominator)
    return (4, 4)


def filter_reasons(score: symusic.Score, fcfg: dict, chill: dict | None = None) -> list[str]:
    reasons = []
    dur = duration_seconds(score)
    tempo = dominant_tempo(score)
    ts = time_signature(score)
    if score.note_num() < fcfg["min_notes"]:
        reasons.append("too_few_notes")
    if not (fcfg["min_duration_sec"] <= dur <= fcfg["max_duration_sec"]):
        reasons.append("duration")
    if not (fcfg["min_tempo"] <= tempo <= fcfg["max_tempo"]):
        reasons.append("tempo")
    if list(ts) not in fcfg["allowed_time_signatures"]:
        reasons.append("time_signature")
    if len(score.tracks) > fcfg["max_tracks"]:
        reasons.append("too_many_tracks")
    if chill is not None:
        if not (chill["min_tempo"] <= tempo <= chill["max_tempo"]):
            reasons.append("chill_tempo")
        if dur > 0 and score.note_num() / dur > chill["max_notes_per_sec"]:
            reasons.append("chill_density")
    return reasons


def _notes_np(track: symusic.Track) -> dict:
    """symusic 0.6 音符容器的 numpy 视图：{time, duration, pitch, velocity}。"""
    return track.notes.numpy()


def _set_notes(track: symusic.Track, time, duration, pitch, velocity) -> None:
    track.notes = track.notes.from_numpy(
        np.ascontiguousarray(time, dtype=np.int32),
        np.ascontiguousarray(duration, dtype=np.int32),
        np.ascontiguousarray(pitch, dtype=np.int8),
        np.ascontiguousarray(velocity, dtype=np.int8),
    )


def estimate_key(score: symusic.Score) -> tuple[int, str]:
    """Krumhansl-Schmuckler：时长加权音级直方图 x 24 调轮廓相关性。"""
    hist = np.zeros(12)
    for track in score.tracks:
        if track.is_drum or track.note_num() == 0:
            continue
        d = _notes_np(track)
        durs = d["duration"].astype(np.float64)
        pcs = d["pitch"] % 12
        for pc in range(12):
            hist[pc] += durs[pcs == pc].sum()
    if hist.sum() == 0:
        return 0, "major"
    hist = hist / hist.sum()
    best, best_key = -2.0, (0, "major")
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if corr > best:
                best, best_key = corr, (tonic, mode)
    return best_key


def map_program(program: int, imap: dict) -> tuple[str, int]:
    for lo, hi, name in PROGRAM_RANGES:
        if lo <= program <= hi:
            return name, imap[name]
    return "piano", imap["piano"]


def normalize(score: symusic.Score, ncfg: dict) -> tuple[symusic.Score, dict]:
    """重采样 -> 量化 -> 乐器映射/选轨 -> 调性归一。返回 (score, info)。"""
    tpq = ncfg["tpq"]
    grid = tpq // ncfg["grid_per_beat"]
    lo, hi = ncfg["pitch_range"]
    score = score.resample(tpq)

    # 量化 + 音域裁剪（逐轨）
    for track in score.tracks:
        if track.note_num() == 0:
            continue
        d = _notes_np(track)
        start = np.round(d["time"] / grid) * grid
        end = np.maximum(np.round((d["time"] + d["duration"]) / grid) * grid, start + grid)
        pitch = d["pitch"]
        if not track.is_drum:
            pitch = np.clip(pitch, lo, hi)
        _set_notes(track, start, end - start, pitch, d["velocity"])

    # 乐器映射 + 按音符数选轨
    kept = []
    for track in score.tracks:
        if track.is_drum:
            kept.append((track.note_num(), track, "percussion"))
            continue
        name, prog = map_program(track.program, ncfg["instrument_map"])
        track.program = prog
        kept.append((track.note_num(), track, name))
    pitched = sorted([k for k in kept if k[2] != "percussion"], key=lambda x: -x[0])
    drums = [k for k in kept if k[2] == "percussion"]
    selected = pitched[: ncfg["keep_top_tracks"]] + drums[:1]
    score.tracks = [k[1] for k in sorted(selected, key=lambda x: -x[0])]

    # 调性归一：大调 -> C，小调 -> a（只移非鼓轨）
    info = {"orig_key": "?", "key_shift": 0}
    if ncfg["key_norm"]:
        tonic, mode = estimate_key(score)
        target = 0 if mode == "major" else 9
        shift = (target - tonic) % 12
        if shift > 6:
            shift -= 12
        if shift != 0:
            for track in score.tracks:
                if not track.is_drum and track.note_num() > 0:
                    d = _notes_np(track)
                    _set_notes(track, d["time"], d["duration"], d["pitch"] + shift, d["velocity"])
        info = {"orig_key": f"{tonic}:{mode}", "key_shift": shift}

    score.sort()
    programs = sorted({("percussion" if t.is_drum else _ensemble_of(t.program, ncfg)) for t in score.tracks})
    info["programs"] = programs
    return score, info


def _ensemble_of(program: int, ncfg: dict) -> str:
    for name, prog in ncfg["instrument_map"].items():
        if prog == program:
            return name
    return "piano"


def transpose_copy(score: symusic.Score, semitones: int, pitch_range: tuple[int, int]) -> symusic.Score | None:
    """移调副本（鼓轨不动）；超出音域则放弃该副本。"""
    lo, hi = pitch_range
    out = score.copy()
    for track in out.tracks:
        if track.is_drum or track.note_num() == 0:
            continue
        d = _notes_np(track)
        p = d["pitch"].astype(np.int32) + semitones
        if p.min() < lo or p.max() > hi:
            return None
        _set_notes(track, d["time"], d["duration"], p, d["velocity"])
    return out


def content_hash(score: symusic.Score, grid: int) -> str:
    items = []
    for track in score.tracks:
        if track.note_num() == 0:
            continue
        d = _notes_np(track)
        prog = -1 if track.is_drum else track.program
        for t, dur, p in zip(d["time"], d["duration"], d["pitch"]):
            items.append((prog, int(p), int(t) // grid, int(dur) // grid))
    items.sort()
    return hashlib.md5(repr(items).encode()).hexdigest()