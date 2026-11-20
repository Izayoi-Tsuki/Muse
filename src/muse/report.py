"""数据集统计与 Markdown 验收报告。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


def _hist(values: list[float], bins: list[float]) -> str:
    if not values:
        return "(空)"
    counts, _ = np.histogram(values, bins=bins)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(c * 40 / max(counts.max(), 1))
        lines.append(f"| {bins[i]:g}–{bins[i + 1]:g} | {c} | {bar} |")
    return "\n".join(lines)


def build_report(
    metas: list[dict],
    reject_stats: dict[str, Counter],
    split_sizes: dict[str, int],
    split_tokens: dict[str, int],
    vocab_size: int,
    dedup_removed: int,
    augment_added: int,
    out_path: Path,
) -> str:
    by_src: dict[str, list[dict]] = {}
    for m in metas:
        by_src.setdefault(m["source"], []).append(m)

    lines = [
        "# M1 数据集统计报告",
        "",
        "## 1. 总览",
        "",
        f"- 入库曲目（基础版，去重后）：**{len(metas)}**",
        f"- 去重剔除：{dedup_removed} 首",
        f"- 移调增广新增序列：{augment_added} 条",
        f"- REMI 词表大小：**{vocab_size}**",
        "",
        "| 来源 | 入库曲目 | 总时长(小时) | 平均音符数 | 平均时长(秒) |",
        "|---|---|---|---|---|",
    ]
    for src, ms in sorted(by_src.items()):
        dur_h = sum(m["duration_sec"] for m in ms) / 3600
        lines.append(
            f"| {src} | {len(ms)} | {dur_h:.2f} | "
            f"{np.mean([m['n_notes'] for m in ms]):.0f} | {np.mean([m['duration_sec'] for m in ms]):.1f} |"
        )

    lines += ["", "## 2. 过滤拒绝原因（按来源）", ""]
    for src, counter in sorted(reject_stats.items()):
        lines.append(f"**{src}**: " + (", ".join(f"{k}×{v}" for k, v in counter.most_common()) or "无拒绝"))
        lines.append("")

    all_durs = [m["duration_sec"] for m in metas]
    all_tempos = [m["tempo"] for m in metas]
    all_lens = [m["token_len"] for m in metas if m["token_len"]]
    lines += [
        "## 3. 分布",
        "",
        "### 时长（秒）",
        "",
        "| 区间 | 数量 | |",
        "|---|---|---|",
        _hist(all_durs, [0, 30, 60, 90, 120, 180, 240, 300, 480]),
        "",
        "### 速度（BPM）",
        "",
        "| 区间 | 数量 | |",
        "|---|---|---|",
        _hist(all_tempos, [40, 60, 80, 100, 120, 140, 160, 180]),
        "",
        "### Token 序列长度（基础版）",
        "",
        f"- 均值 {np.mean(all_lens):.0f} / 中位 {np.median(all_lens):.0f} / "
        f"最大 {max(all_lens)} / 最小 {min(all_lens)}" if all_lens else "(空)",
        "",
        "### 原始调性分布（归一前）",
        "",
    ]
    key_counter = Counter(m["orig_key"] for m in metas)
    lines.append(", ".join(f"{k}×{v}" for k, v in key_counter.most_common(12)))
    lines += ["", "### 乐器编制出现率（映射后）", ""]
    prog_counter = Counter(p for m in metas for p in m["programs"])
    for name, c in prog_counter.most_common():
        lines.append(f"- {name}: {c} 首 ({c * 100 // max(len(metas), 1)}%)")

    lines += [
        "",
        "## 4. 数据集划分（按曲目整体划分，增广副本随原曲）",
        "",
        "| 划分 | 序列数 | Token 总量 |",
        "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {split_sizes.get(split, 0)} | {split_tokens.get(split, 0):,} |")

    lines += [
        "",
        "## 5. 说明",
        "",
        "- 量化：1/16 拍（480 tpq，网格 120 tick）；调性归一：Krumhansl 估调后大调→C、小调→a。",
        "- 记谱修正：POP909 等数据集中 1/4、2/4、2/2 拍号伪影统一重写为 4/4（不改音符时序）。",
        "- 乐器映射：钢琴/风琴/合成→钢琴，吉他族→尼龙吉他，贝斯族→原声贝斯，色彩打击/民族→马林巴，弦乐/合唱/管乐→弦乐合奏，鼓轨保留。",
        "- 增广：动森与 POP909 每首额外 2 个 ±3 半音移调副本（确定性种子）；Lakh 不增广。",
        "- Lakh 经休闲风格过滤（55–140 BPM、密度≤12 音符/秒）并限量。",
        "- 所有序列含 BOS/EOS；按小节切分编码（encode_ids_split=bar），便于 M3 滑窗续写。",
    ]
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text