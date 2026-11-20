"""M2 训练：加权混合预训练 -> 风格微调，bf16 + 余弦调度 + 断点续训。

用法:
  python scripts/train.py                 # 完整两阶段
  python scripts/train.py --resume checkpoints/step_xxxx.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.model import MuseTransformer  # noqa: E402


def load_chunks(tok_dir: Path, split: str, seq_len: int, sources: list[str] | None) -> dict[str, np.ndarray]:
    """把每个来源的 token 流拼接后切成 seq_len+1 的块（x=[:-1], y=[1:]）。"""
    streams: dict[str, list[int]] = {}
    with open(tok_dir / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if sources is not None and r["source"] not in sources:
                continue
            streams.setdefault(r["source"], []).extend(r["ids"])
    out = {}
    for src, ids in streams.items():
        arr = np.asarray(ids, dtype=np.int32)
        n = len(arr) // (seq_len + 1)  # 无重叠切块，每块 seq_len+1（x/y 错位 1）
        if n > 0:
            out[src] = arr[: n * (seq_len + 1)].reshape(n, seq_len + 1)
    return out


class WeightedChunks(torch.utils.data.Dataset):
    """按来源权重采样 chunk；权重作用于来源选择概率。"""

    def __init__(self, chunks: dict[str, np.ndarray], weights: dict[str, float], length: int):
        self.chunks = chunks
        self.sources = list(chunks.keys())
        w = np.array([weights.get(s, 1.0) for s in self.sources], dtype=np.float64)
        self.probs = w / w.sum()
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        src = self.sources[np.random.choice(len(self.sources), p=self.probs)]
        bank = self.chunks[src]
        return torch.from_numpy(bank[random.randrange(len(bank))].copy())


def flat_loader(chunks: dict[str, np.ndarray], batch_size: int, shuffle: bool):
    arr = np.concatenate(list(chunks.values()), axis=0) if chunks else np.zeros((0, 1), np.int32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(arr))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def lr_at(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, loader, device, dtype, max_batches: int = 40) -> float:
    model.eval()
    losses = []
    for i, (batch,) in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def run_phase(
    name: str,
    model: MuseTransformer,
    train_chunks: dict,
    val_loader,
    tcfg: dict,
    phase: dict,
    device: torch.device,
    dtype: torch.dtype,
    ckpt_dir: Path,
    log_path: Path,
    start_step: int = 0,
    optimizer_state: dict | None = None,
) -> int:
    weights = tcfg.get("source_weights", {})
    steps_per_epoch = max(1, sum(len(c) for c in train_chunks.values()) // tcfg["batch_size"])
    total_steps = steps_per_epoch * phase["epochs"]
    ds = WeightedChunks(train_chunks, weights, length=total_steps * tcfg["batch_size"])
    loader = torch.utils.data.DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(
        model.parameters(), lr=phase["lr"], betas=tuple(tcfg["betas"]), weight_decay=tcfg["weight_decay"]
    )
    if optimizer_state:
        opt.load_state_dict(optimizer_state)
    scaler_enabled = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    model.train()
    step = start_step
    t0 = time.time()
    log_f = open(log_path, "a", encoding="utf-8")
    for batch in loader:
        batch = batch.to(device=device, dtype=torch.long)
        lr = lr_at(step - start_step, phase["warmup_steps"], total_steps, phase["lr"], phase["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % tcfg["log_every"] == 0:
            dt = time.time() - t0
            msg = (f"[{name}] step {step}/{start_step + total_steps} loss {loss.item():.4f} "
                   f"lr {lr:.2e} {dt / tcfg['log_every']:.2f}s/it")
            print(msg, flush=True)
            log_f.write(json.dumps({"phase": name, "step": step, "loss": loss.item(), "lr": lr}) + "\n")
            log_f.flush()
            t0 = time.time()
        if step % tcfg["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device, dtype)
            msg = f"[{name}] step {step} VAL loss {val_loss:.4f} ppl {math.exp(min(val_loss, 20)):.1f}"
            print(msg, flush=True)
            log_f.write(json.dumps({"phase": name, "step": step, "val_loss": val_loss}) + "\n")
            log_f.flush()
        if step % tcfg["save_every"] == 0:
            save_checkpoint(ckpt_dir / f"{name}_step_{step}.pt", model, opt, step, model.cfg)
    save_checkpoint(ckpt_dir / f"{name}_final.pt", model, opt, step, model.cfg)
    log_f.close()
    return step


def save_checkpoint(path: Path, model, opt, step: int, mcfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "model_cfg": mcfg}, path)
    print(f"saved {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg, mcfg = cfg["train"], cfg["model"]
    torch.manual_seed(tcfg["seed"])
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    device = torch.device("cuda")
    dtype = torch.bfloat16 if tcfg["amp_dtype"] == "bfloat16" else torch.float16

    tok_dir = ROOT / cfg["paths"]["tokenized"]
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    log_dir = ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)

    # 词表一致性检查
    vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    if len(vocab) != mcfg["vocab_size"]:
        mcfg["vocab_size"] = len(vocab)
        print(f"vocab_size adjusted to {len(vocab)}")

    print("loading chunks...", flush=True)
    train_chunks = load_chunks(tok_dir, "train", mcfg["max_seq_len"], None)
    val_chunks = load_chunks(tok_dir, "val", mcfg["max_seq_len"], None)
    for s, c in train_chunks.items():
        print(f"  train {s}: {len(c)} chunks ({len(c) * mcfg['max_seq_len'] / 1e6:.1f}M tokens)")
    val_loader = flat_loader(val_chunks, tcfg["batch_size"], shuffle=False)

    model = MuseTransformer(mcfg).to(device)
    print(f"model params: {model.num_params() / 1e6:.1f}M", flush=True)

    start_step, opt_state = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        start_step, opt_state = ck["step"], ck["optimizer"]
        print(f"resumed from {args.resume} @ step {start_step}")

    log_path = log_dir / "train.jsonl"
    step = run_phase("pretrain", model, train_chunks, val_loader, tcfg, tcfg, device, dtype,
                     ckpt_dir, log_path, start_step, opt_state)

    ft = cfg.get("finetune", {})
    if ft.get("enabled"):
        ft_chunks = load_chunks(tok_dir, "train", mcfg["max_seq_len"], ft["sources"])
        if ft_chunks:
            print(f"finetune on {list(ft_chunks.keys())}", flush=True)
            run_phase("finetune", model, ft_chunks, val_loader, tcfg, ft, device, dtype,
                      ckpt_dir, log_path, step)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()