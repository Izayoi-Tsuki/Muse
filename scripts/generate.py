"""M2 采样生成：加载检查点 -> 续写 token -> 解码 MIDI ->（可选）渲染音频。

用法:
  python scripts/generate.py --checkpoint checkpoints/finetune_final.pt --tokens 4096
  python scripts/generate.py --checkpoint ... --prompt-from-data --n-samples 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from miditok import REMI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.model import MuseTransformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--prompt-from-data", action="store_true", help="用真实曲目前 64 token 做引子")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MuseTransformer(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.checkpoint} (step {ck['step']}, {model.num_params() / 1e6:.1f}M params)")

    tok = REMI(params=ROOT / "data/tokenized/tokenizer.json")
    pad_id, bos_id = 0, 1
    prompts: list[list[int]] = []
    if args.prompt_from_data:
        records = []
        with open(ROOT / "data/tokenized/train.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["source"] == "animal_crossing" and r["shift"] == 0:
                    records.append(r)
        rng = random.Random(args.seed)
        for r in rng.sample(records, min(args.n_samples, len(records))):
            prompts.append(r["ids"][:64])
            print(f"prompt from {r['song_id']}")
    while len(prompts) < args.n_samples:
        prompts.append([bos_id])

    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)
    for i, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + i)
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        gen = model.generate(idx, args.tokens, args.temperature, args.top_p)
        ids = [t for t in gen[0].tolist() if t != pad_id]
        score = tok.decode(ids)
        mid_path = out_dir / f"gen_{args.seed + i:03d}.mid"
        score.dump_midi(str(mid_path))
        dur = score.end() / max(score.tpq, 1)
        print(f"sample {i}: {score.note_num()} notes, {len(score.tracks)} tracks -> {mid_path}")
        if args.render:
            from render_midi import render  # noqa: E402
            wav = render(mid_path)
            if wav:
                print(f"  rendered -> {wav}")


if __name__ == "__main__":
    main()