"""Benchmark MiniMind training throughput on a real device, per experimental arm.

bench_m5.py already exists but only sweeps three fixed (hidden_size, num_hidden_layers)
configs on whatever local device is available; it has no vocab_size axis (both arms'
vocab sizes matter for embedding/lm_head FLOPs and memory), no --device selection (the
rented-GPU decision in 决策-20260918.md §2.2 needs a number measured ON that GPU, not
extrapolated from M5), and no batch-size sweep (throughput vs. batch size is exactly
what "两臂必须同 batch"（§1.5）needs to be chosen from). Hence a new, separate script
rather than editing bench_m5.py.

"Effective TFLOPS" uses the same nominal per-position FLOPs formula as the training
ledger (决策文档 §1.4(b): flops_nominal_per_position = 6*N + 12*L*H*S), not the naive
6*N*tokens approximation bench_m5.py uses -- so a bench_gpu.py number and a run_ledger
flops_nominal_cum number are directly comparable.

The benchmarked step must be the SAME arithmetic the patched trainer runs, or its two
outputs are both actively misleading. §1.5 picks the production batch size from this
script's memory numbers and 风险表 #2 gates the whole rental on its tok/s. An earlier
version cast the whole model (and hence AdamW's moments) to bf16, while training keeps
fp32 master weights under torch.autocast: that halved the reported memory (arm A, batch
8: 397 MB vs 793 MB) and measured pure-bf16 throughput, i.e. it would have recommended
a batch size that OOMs in the real run. So bench_one now mirrors the trainer exactly --
fp32 parameters, autocast only on cuda, GradScaler enabled only for float16, and the
same clip_grad_norm_ -- and reports precision_regime so the JSON says which regime the
numbers came from. On mps/cpu there is no autocast in either program, so --dtype is
recorded as requested but the run really is fp32; that is stated in the output instead
of being quietly assumed.

Memory on mps needs the same care. torch.mps has no max_memory_allocated, and sampling
current_allocated_memory once per step after optimizer.zero_grad() samples the trough
of the step, not its peak: measured on arm A / batch 8 / fp32, that point reads 793.9 MB
while the same step peaks at 4058.9 MB right after the forward. This script samples
after the forward and after the backward and keeps the max, and also reports the driver
high-water mark (torch.mps.driver_allocated_memory), which is the number that actually
has to fit.
"""
import argparse
from contextlib import nullcontext
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import optim

from common import ROOT, sha256_file, write_json

sys.path.insert(0, str(ROOT / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

DTYPES = {"fp32": torch.float32, "float32": torch.float32,
          "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
          "fp16": torch.float16, "float16": torch.float16}

# Measured on this project's own configs with 5 seeds, random and real packed tokens,
# on code paths that do not touch the trainer patch. A freshly initialized model does
# NOT score ln(vocab_size): HF's initializer_range=0.02 normal init is not the uniform
# distribution that ln(V) assumes, and E[-log p] sits systematically above it.
# 决策文档 风险表 #5 would kill a perfectly healthy run on "step-0 loss != ln(V)±0.05".
STEP0_BASELINE = {6400: 8.91, 16384: 9.86}  # +-0.05, hidden_size=768


def flops_nominal_per_position(model, num_hidden_layers, hidden_size, seq_len):
    """Duplicated from minimind/trainer/trainer_utils.py (lab/upstream-pretrain.patch).

    Kept as a second, independent copy on purpose: this script must be usable to
    bench a device BEFORE the patch is applied (e.g. to decide whether renting a
    given GPU is worth it at all), so it cannot import from the patched trainer.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return 6 * n_params + 12 * num_hidden_layers * hidden_size * seq_len


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def is_oom(exc):
    """True for 'this batch size does not fit', false for every other RuntimeError.

    A blanket `except RuntimeError` would file a genuine modelling bug as a memory
    limit and keep sweeping, which is exactly the silent failure 风险表 #2 must not
    have in its own gate.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return any(signal in text for signal in
               ("out of memory", "invalid buffer size", "mps backend out of memory",
                "can't allocate", "cannot allocate"))


def bench_one(cfg, seq_len, batch, steps, warmup, device, dtype, grad_clip, compile_model):
    device_type = "cuda" if "cuda" in device else "cpu"  # same test the trainer makes
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=dtype)
                    if device_type == "cuda" else nullcontext())
    effective_dtype = str(dtype).replace("torch.", "") if device_type == "cuda" else "float32"
    # fp32 parameters, exactly like the trainer: autocast keeps master weights in
    # fp32 and AdamW therefore carries fp32 moments. Casting the model itself would
    # measure a different program.
    model = MiniMindForCausalLM(cfg).to(device)
    if compile_model:
        model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)

    # Sanity check 1 from bench_m5.py's own playbook: a fresh random init must score
    # close to a KNOWN baseline for this config. Computed in fp32 without autocast so
    # the number is a property of the init, not of the precision regime.
    model.eval()
    with torch.no_grad():
        step0_loss = model(ids, labels=ids).loss.item()
    model.train()

    def one_step(sample_memory=False):
        peak = 0
        with autocast_ctx:
            out = model(ids, labels=ids)
            loss = out.loss + out.aux_loss
        if sample_memory:
            peak = max(peak, torch.mps.current_allocated_memory())
        scaler.scale(loss).backward()
        if sample_memory:
            peak = max(peak, torch.mps.current_allocated_memory())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return peak

    for _ in range(warmup):
        one_step()
    synchronize(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    peak_mps_bytes = 0
    sample = device == "mps"

    started = time.perf_counter()
    for _ in range(steps):
        peak_mps_bytes = max(peak_mps_bytes, one_step(sample_memory=sample))
    synchronize(device)
    elapsed = time.perf_counter() - started

    driver_bytes = None
    if device.startswith("cuda"):
        peak_bytes = torch.cuda.max_memory_allocated(device)
    elif device == "mps":
        peak_bytes = peak_mps_bytes
        driver_bytes = torch.mps.driver_allocated_memory()
    else:
        peak_bytes = None

    flops_per_position = flops_nominal_per_position(model, cfg.num_hidden_layers, cfg.hidden_size, seq_len)
    positions = batch * seq_len * steps
    baseline = STEP0_BASELINE.get(cfg.vocab_size)
    result = {
        "batch": batch, "seq_len": seq_len, "steps": steps, "warmup": warmup,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision_regime": ("autocast(cuda, %s) over fp32 master weights" % effective_dtype
                             if device_type == "cuda" else
                             "fp32 throughout (no autocast on this device; --dtype ignored)"),
        "effective_dtype": effective_dtype,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "step0_loss_nats": step0_loss, "step0_loss_regime": "fp32, no autocast",
        "ln_vocab_size": math.log(cfg.vocab_size),
        "step0_loss_minus_ln_vocab": step0_loss - math.log(cfg.vocab_size),
        "step0_baseline_nats": baseline,
        "step0_within_baseline": None if baseline is None else abs(step0_loss - baseline) <= 0.05,
        "elapsed_seconds": elapsed, "ms_per_step": elapsed / steps * 1000,
        "tokens_per_second": positions / elapsed,
        "flops_nominal_per_position": flops_per_position,
        "effective_tflops": flops_per_position * positions / elapsed / 1e12,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_source": ("torch.cuda.max_memory_allocated" if device.startswith("cuda")
                               else "max(torch.mps.current_allocated_memory) after fwd and bwd"
                               if device == "mps" else None),
        "driver_allocated_bytes": driver_bytes,
    }
    del model, optimizer, scaler
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--batch-sweep", default="32,64,128", help="comma-separated batch sizes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=512, help="must match the packed sequence length")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32",
                        help="只在 cuda 上生效（与训练补丁一致）；mps/cpu 上实际跑 fp32")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="与训练一致的梯度裁剪阈值")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    batch_sizes = [int(token) for token in args.batch_sweep.split(",") if token.strip()]
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError(f"--batch-sweep must be a nonempty list of positive ints, got {args.batch_sweep!r}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    arm_config = json.loads((ROOT / "lab/configs/arms.json").read_text())[args.arm]
    tokenizer_dir = ROOT / arm_config["tokenizer"]
    lm_config = MiniMindConfig(hidden_size=arm_config["hidden_size"],
                               num_hidden_layers=arm_config["num_hidden_layers"],
                               vocab_size=arm_config["vocab_size"])
    dtype = DTYPES[args.dtype]

    print(f"device={args.device} arm={args.arm} dtype={args.dtype} compile={args.compile} "
          f"vocab_size={lm_config.vocab_size} hidden={lm_config.hidden_size} "
          f"layers={lm_config.num_hidden_layers} torch={torch.__version__}", flush=True)

    runs = []
    for batch in batch_sizes:
        print(f"batch={batch} ...", flush=True)
        try:
            row = bench_one(lm_config, args.seq_len, batch, args.steps, args.warmup,
                            args.device, dtype, args.grad_clip, args.compile)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            row = {"batch": batch, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED (out of memory): {row['error']}", flush=True)
        else:
            memory = "" if row["peak_memory_bytes"] is None else \
                f" | {row['peak_memory_bytes'] / 2**20:7.0f} MiB peak"
            print(f"  {row['ms_per_step']:7.1f} ms/step | {row['tokens_per_second']:9.0f} tok/s | "
                  f"{row['effective_tflops']:6.2f} eff.TFLOPS{memory} | "
                  f"step0_loss={row['step0_loss_nats']:.3f} "
                  f"(ln V={row['ln_vocab_size']:.3f}, baseline={row['step0_baseline_nats']})", flush=True)
        runs.append(row)

    ok = [row for row in runs if "error" not in row]
    result = {
        "arm": args.arm, "device": args.device, "requested_dtype": args.dtype,
        "precision_regime": ok[0]["precision_regime"] if ok else None,
        "compile": args.compile, "grad_clip": args.grad_clip,
        "seq_len": args.seq_len, "steps": args.steps, "warmup": args.warmup,
        "arm_config": arm_config, "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "torch": torch.__version__, "runs": runs,
        "note": "throughput of the compute step only; the packed-memmap read path is not "
                "included. Re-check against a real --profile pack run on the rented GPU.",
    }
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok:
        # 风险表 #2 uses this script as the rent-or-not gate; `bench_gpu.py && train`
        # must not proceed when every batch size failed.
        print("every batch size failed -- no usable throughput measurement", file=sys.stderr)
        sys.exit(1)
