"""M1 主流程：收集 -> 过滤 -> 规范化 -> 去重 -> REMI token 化 -> 划分 -> 报告。

用法: python scripts/build_dataset.py [--config configs/data.yaml] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse import midi_utils as mu  # noqa: E402
from muse import report as rp  # noqa: E402
from muse import tokenize as tk  # noqa: E402


def iter_source_files(raw: Path, scfg: dict) -> list[Path]:
    files = sorted(raw.glob(scfg["glob"]))
    if scfg.get("max_files"):
        rng = random.Random(scfg.get("seed", 42))
        files = list(files)
        rng.shuffle(files)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="每来源最多处理文件数（调试用）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw = ROOT / cfg["paths"]["raw"]
    interim = ROOT / cfg["paths"]["interim"]
    tok_dir = ROOT / cfg["paths"]["tokenized"]
    tok_dir.mkdir(parents=True, exist_ok=True)
    fcfg, ncfg = cfg["filter"], cfg["normalize"]
    grid = ncfg["tpq"] // ncfg["grid_per_beat"]

    tokenizer = tk.build_tokenizer(cfg["tokenizer"])
    print(f"tokenizer: REMI, vocab={len(tokenizer)}")

    metas: list[dict] = []
    records: list[dict] = []
    reject_stats: dict[str, Counter] = {}
    seen_hashes: set[str] = set()
    dedup_removed = 0
    augment_added = 0

    for source, scfg in cfg["sources"].items():
        files = iter_source_files(raw, scfg)
        if args.limit:
            files = files[: args.limit]
        chill = fcfg.get("lakh_chill") if source == "lakh" else None
        accepted = 0
        rej = reject_stats.setdefault(source, Counter())
        print(f"\n=== {source}: {len(files)} candidates ===")
        for i, path in enumerate(files):
            if scfg.get("max_files") and accepted >= scfg["max_files"]:
                break
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} scanned, {accepted} accepted", flush=True)
            score = mu.load_score(path)
            if score is None:
                rej["unreadable"] += 1
                continue
            reasons = mu.filter_reasons(score, fcfg, chill)
            if reasons:
                for r in reasons:
                    rej[r] += 1
                continue
            try:
                score, info = mu.normalize(score, ncfg)
            except Exception as e:
                rej[f"normalize_error:{type(e).__name__}"] += 1
                continue
            h = mu.content_hash(score, grid)
            if h in seen_hashes:
                dedup_removed += 1
                continue
            seen_hashes.add(h)

            song_id = f"{scfg['tag']}:{path.stem}"
            dur = mu.duration_seconds(score)
            meta = {
                "song_id": song_id,
                "source": source,
                "path": str(path.relative_to(raw)),
                "n_notes": score.note_num(),
                "duration_sec": round(dur, 2),
                "tempo": round(mu.dominant_tempo(score), 1),
                "time_signature": list(mu.time_signature(score)),
                "orig_key": info["orig_key"],
                "key_shift": info["key_shift"],
                "programs": info["programs"],
                "n_tracks": len(score.tracks),
                "token_len": 0,
            }
            try:
                base_ids = tk.encode(tokenizer, score)
            except Exception as e:
                rej[f"tokenize_error:{type(e).__name__}"] += 1
                continue
            meta["token_len"] = len(base_ids)
            metas.append(meta)
            records.append({"ids": base_ids, "song_id": song_id, "source": source, "shift": 0})

            # 保存清洗后的基础 MIDI（供后续渲染/检查）
            midi_out = interim / source / f"{path.stem}.mid"
            midi_out.parent.mkdir(parents=True, exist_ok=True)
            score.dump_midi(str(midi_out))

            # 移调增广
            if scfg.get("augment"):
                for shift, variant in tk.augmented_variants(score, cfg["augment"], song_id, tuple(ncfg["pitch_range"])):
                    try:
                        ids = tk.encode(tokenizer, variant)
                    except Exception:
                        continue
                    records.append({"ids": ids, "song_id": song_id, "source": source, "shift": shift})
                    augment_added += 1
            accepted += 1
        print(f"  accepted {accepted}, rejects: {dict(rej)}")

    # 按曲目整体划分
    scfg = cfg["split"]
    songs = sorted({r["song_id"] for r in records})
    random.Random(scfg["seed"]).shuffle(songs)
    n_val = max(1, int(len(songs) * scfg["val_ratio"]))
    n_test = max(1, int(len(songs) * scfg["test_ratio"]))
    split_of = {}
    for i, s in enumerate(songs):
        split_of[s] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")

    split_sizes, split_tokens = Counter(), Counter()
    writers = {}
    try:
        for split in ("train", "val", "test"):
            writers[split] = open(tok_dir / f"{split}.jsonl", "w", encoding="utf-8")
        for r in records:
            split = split_of[r["song_id"]]
            writers[split].write(json.dumps({"ids": r["ids"], "song_id": r["song_id"],
                                             "source": r["source"], "shift": r["shift"]}) + "\n")
            split_sizes[split] += 1
            split_tokens[split] += len(r["ids"])
    finally:
        for w in writers.values():
            w.close()

    # 保存 tokenizer 参数与词表
    tokenizer.save(tok_dir / "tokenizer.json")
    (tok_dir / "vocab.json").write_text(
        json.dumps({str(k): v for k, v in enumerate(tokenizer.vocab)}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    (tok_dir / "manifest.json").write_text(
        json.dumps({"config": cfg, "n_songs": len(metas), "n_sequences": len(records),
                    "vocab_size": len(tokenizer), "split_sizes": dict(split_sizes)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text = rp.build_report(
        metas, reject_stats, dict(split_sizes), dict(split_tokens),
        len(tokenizer), dedup_removed, augment_added,
        ROOT / cfg["paths"]["reports"] / "dataset_report.md",
    )
    print("\n" + text)


if __name__ == "__main__":
    main()