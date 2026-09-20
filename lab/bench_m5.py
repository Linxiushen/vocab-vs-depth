"""实测 M5 上 MiniMind 的训练吞吐，用来校准所有工期估算。
用法: python lab/bench_m5.py [--steps 30]
"""
import argparse, math, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minimind'))
import torch
from torch import optim
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

CONFIGS = [
    ("smoke  8L/512",  dict(hidden_size=512,  num_hidden_layers=8),  8, 512),
    ("复现档 8L/768",  dict(hidden_size=768,  num_hidden_layers=8),  8, 768),
    ("推荐档 16L/768", dict(hidden_size=768,  num_hidden_layers=16), 8, 768),
]

def bench(name, cfg_kw, batch, seq, steps, device, dtype):
    cfg = MiniMindConfig(**cfg_kw)
    model = MiniMindForCausalLM(cfg).to(device=device, dtype=dtype)
    N = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=5e-4)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    # 判据 1: step-0 loss 必须 ≈ ln(vocab_size)
    model.eval()
    with torch.no_grad():
        l0 = model(ids, labels=ids).loss.item()
    model.train()

    for _ in range(3):  # warmup
        loss = model(ids, labels=ids).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device == "mps": torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        loss = model(ids, labels=ids).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device == "mps": torch.mps.synchronize()
    dt = time.perf_counter() - t0

    toks = batch * seq * steps
    tps = toks / dt
    tflops = 6 * N * toks / dt / 1e12   # 6·N·D，含 fwd+bwd
    mem = torch.mps.current_allocated_memory()/1e9 if device == "mps" else 0
    print(f"{name:16s} N={N/1e6:6.2f}M b={batch} seq={seq} dtype={str(dtype).split('.')[-1]:8s}"
          f" | step0_loss={l0:.3f} (期望 {math.log(cfg.vocab_size):.3f})"
          f" | {dt/steps*1000:7.1f} ms/step | {tps:8.0f} tok/s | {tflops:5.2f} TFLOPS | mem {mem:.2f}GB")
    del model, opt
    if device == "mps": torch.mps.empty_cache()
    return tps, tflops

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}\n")
    for dtype in (torch.float32, torch.bfloat16):
        print(f"--- {str(dtype).split('.')[-1]} ---")
        for name, kw, b, s in CONFIGS:
            try:
                bench(name, kw, b, s, a.steps, dev, dtype)
            except Exception as e:
                print(f"{name:16s} FAILED: {type(e).__name__}: {str(e)[:120]}")
        print()
