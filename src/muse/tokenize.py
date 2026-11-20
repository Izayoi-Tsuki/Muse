"""REMI token 化：tokenizer 构建、移调增广、序列编码。"""
from __future__ import annotations

import random

import symusic
from miditok import REMI, TokenizerConfig

from .midi_utils import transpose_copy


def build_tokenizer(tcfg: dict) -> REMI:
    # beat_res {(0,4):4} = 每拍 4 个位置 = 1/16 拍分辨率，与量化网格一致
    config = TokenizerConfig(
        pitch_range=(21, 108),
        beat_res={(0, 4): 4},
        num_velocities=tcfg["num_velocities"],
        special_tokens=list(tcfg["special_tokens"]),
        encode_ids_split="bar",
        use_velocities=True,
        use_programs=True,
    )
    return REMI(config)


def encode(tokenizer: REMI, score: symusic.Score) -> list[int]:
    seq = tokenizer(score)
    return list(seq.ids)


def augmented_variants(
    score: symusic.Score,
    acfg: dict,
    song_id: str,
    pitch_range: tuple[int, int],
) -> list[tuple[int, symusic.Score]]:
    """确定性移调增广：同一 song_id 永远得到相同副本。返回 [(shift, score)]。"""
    rng = random.Random(f"{acfg['seed']}:{song_id}")
    lo, hi = acfg["transpose_range"]
    choices = [s for s in range(lo, hi + 1) if s != 0]
    shifts = rng.sample(choices, min(acfg["copies"], len(choices)))
    out = []
    for s in shifts:
        variant = transpose_copy(score, s, pitch_range)
        if variant is not None:
            out.append((s, variant))
    return out