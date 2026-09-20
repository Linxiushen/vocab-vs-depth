"""Run the upstream pretrainer with frozen experiment arms and isolated outputs.

The default check profile only verifies forward/backward/save on 16 documents.
The full profile runs one epoch over the frozen training corpus (legacy, padded,
epoch-based -- kept only for backward compatibility with earlier runs).
The pack profile is the one described in 决策-20260918.md §1.5: fixed-token-budget
training over lab/data/v3/packed_{arm} (produced by lab/pack_corpus.py), with the
WSD/cosine schedule and per-update ledger implemented in the trainer patch.

Why the CLI is strict here. §1.5's command is meant to be copy-pasted verbatim,
and it passes --packed without --profile. An earlier version of this file spelled
the flag --packed_prefix and left argparse's allow_abbrev on, so `--packed <path>`
was prefix-matched into it, the profile stayed at its "check" default, and every
pack-only flag (--total_tokens, --batch_size 64, --schedule, --dtype, ...) was
silently dropped: the documented production command ran 8 micro-steps on 16
documents at batch 2 / seq 128 and reported status=complete. Hence three rules,
all of which have to hold together for that failure to be impossible:
  1. allow_abbrev=False -- no flag is ever absorbed by a longer one;
  2. the flag is named --packed, exactly as interface contract P3 says;
  3. every pack-only argument defaults to None, so "was it given?" is knowable:
     giving one under check/full is a hard error instead of a silent no-op, and
     giving --packed with no --profile selects the pack profile rather than
     quietly running a different experiment.
"""
import argparse
from itertools import islice
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from common import ROOT, read_texts, sha256_file, write_json, write_texts


# Defaults for the pack profile. They live here rather than in add_argument()
# because an argparse default is indistinguishable from a value the user typed,
# and rule 3 above depends on telling those two apart.
PACK_DEFAULTS = {
    "packed": None,  # None -> lab/data/v3/packed_{arm}
    "total_tokens": None,  # required
    "batch_size": 64,
    "accumulation_steps": 1,
    "max_seq_len": 512,
    "learning_rate": 2e-4,
    "schedule": "wsd",
    "warmup_frac": 0.02,
    "decay_frac": 0.10,
    "lr_floor": 0.1,
    "grad_clip": 1.0,
    "dtype": "bf16",
    "save_steps_frac": 0.05,
    "log_interval": 10,
    "keep_last_n": 0,
    "resume": 0,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--profile", choices=["check", "full", "pack"], default=None,
                        help="default: pack when --packed is given, otherwise check")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    # ---- pack profile only (决策文档 §1.5). All default to None; see module docstring.
    pack_only = parser.add_argument_group(
        "pack profile", "只有 --profile pack 接受这些参数；check/full 的 batch/seq/lr 是写死的")
    pack_only.add_argument("--packed", "--packed_prefix", default=None,
                           help=f"packed 数据前缀，默认 lab/data/v3/packed_{{arm}}")
    pack_only.add_argument("--total_tokens", type=float, default=None,
                           help="监督 token 预算；见决策文档 §1.4/§1.5")
    pack_only.add_argument("--batch_size", type=int, default=None)
    pack_only.add_argument("--accumulation_steps", type=int, default=None)
    pack_only.add_argument("--max_seq_len", type=int, default=None)
    pack_only.add_argument("--learning_rate", type=float, default=None)
    pack_only.add_argument("--schedule", choices=["cosine", "wsd"], default=None)
    pack_only.add_argument("--warmup_frac", type=float, default=None)
    pack_only.add_argument("--decay_frac", type=float, default=None)
    pack_only.add_argument("--lr_floor", type=float, default=None)
    pack_only.add_argument("--grad_clip", type=float, default=None)
    pack_only.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16",
                                               "fp32", "float32"], default=None)
    pack_only.add_argument("--save_steps_frac", type=float, default=None)
    pack_only.add_argument("--log_interval", type=int, default=None)
    pack_only.add_argument("--keep_last_n", type=int, default=None,
                           help="最多保留多少个中途 checkpoint（0=全留，终点永不删）")
    pack_only.add_argument("--resume", type=int, default=None, choices=[0, 1],
                           help="从 <out>/weights/pretrain_<hidden>_resume.pt 续训（fp32 权重+optimizer+全部累计量）")
    args = parser.parse_args()

    profile = args.profile or ("pack" if args.packed is not None else "check")
    given = sorted(name for name in PACK_DEFAULTS if getattr(args, name) is not None)
    if profile != "pack" and given:
        parser.error(f"--profile {profile} 不接受 pack 专用参数 {given}："
                     f"check/full 的 batch/seq/lr/日程都是写死的，静默忽略它们会让一条"
                     f"看起来在跑正式实验的命令跑成 16 篇文档的冒烟测试。要跑正式训练请加 --profile pack")
    for name, default in PACK_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    arm = json.loads((ROOT / "lab/configs/arms.json").read_text())[args.arm]
    tokenizer = ROOT / arm.pop("tokenizer")
    if not (tokenizer / "training_manifest.json").is_file():
        parser.error(f"Train tokenizer first: {tokenizer}")
    tokenizer_sha256 = sha256_file(tokenizer / "tokenizer.json")

    # Everything that can reject this invocation is checked BEFORE the run
    # directory exists, so a rejected command leaves no half-built lab/runs/ entry.
    extra_manifest = {}
    if profile == "pack":
        if args.total_tokens is None or args.total_tokens <= 0:
            parser.error("--profile pack requires --total_tokens > 0")
        # Resolved to an absolute path because the trainer subprocess runs with
        # cwd=work/, not the project root. A relative --packed passed straight
        # through would pass the launcher's own is_file() check here and then
        # fail inside the child with FileNotFoundError after the model was built.
        prefix = (Path(args.packed).resolve() if args.packed
                  else ROOT / f"lab/data/v3/packed_{args.arm}")
        meta_path = Path(f"{prefix}.meta.json")
        if not meta_path.is_file():
            parser.error(f"packed data missing, run lab/pack_corpus.py --arm {args.arm} first: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta["seq_len"]) != args.max_seq_len:
            parser.error(f"packed seq_len {meta['seq_len']} != --max_seq_len {args.max_seq_len}")
        if int(meta["vocab_size"]) != int(arm["vocab_size"]):
            parser.error(f"packed vocab_size {meta['vocab_size']} != arm {args.arm} vocab_size {arm['vocab_size']}")
        # vocab_size is not an identity: the upstream model/ tokenizer (71f32c68…)
        # and lab/tokenizers/bpe_6400 (cdd7e5bc…) are both 6400 wide with different
        # merges, and pairing the wrong one with these ids trains a checkpoint that
        # loads fine and emits garbage (决策文档 风险表 #4). The trainer patch
        # repeats this check; doing it here too means the launcher never even
        # starts a run that is doomed. Recording the hash without comparing it --
        # what this file used to do -- is not a check.
        if meta.get("tokenizer_sha256") != tokenizer_sha256:
            parser.error(f"arm {args.arm} tokenizer {tokenizer} has tokenizer.json sha256 "
                         f"{tokenizer_sha256}, but {meta_path.name} was packed with "
                         f"{meta.get('tokenizer_sha256')}: repack, or point --packed at the "
                         f"other arm's data")
        targets_per_seq = int(meta["seq_len"]) - 1
        tokens_per_update = args.batch_size * targets_per_seq * args.accumulation_steps
        total_tokens = int(args.total_tokens)
        total_updates = math.ceil(total_tokens / tokens_per_update)
        extra_manifest = {
            "packed_prefix": str(prefix),
            "packed_meta_sha256": sha256_file(meta_path),
            "packed_ids_sha256": sha256_file(f"{prefix}.ids.u16"),
            # P1 identity fields, recorded so a finished run can prove -- without
            # the packed data still being around -- that both arms walked the same
            # doc_order and that the ids came from the arm's own tokenizer.
            "packed_tokenizer_sha256": meta.get("tokenizer_sha256"),
            "packed_doc_order_sha256": meta.get("doc_order_sha256"),
            "packed_n_seq": int(meta["n_seq"]),
            "tokens_per_update": tokens_per_update,
            "total_updates": total_updates,
            "total_tokens_requested": total_tokens,
        }

    run = args.out.resolve()
    if args.resume:
        # Resuming into the SAME directory on purpose: the trainer's resume state,
        # the inference checkpoints and the ledger it truncates all live here, and
        # the run is only meaningful as one record. A fresh directory would give a
        # second manifest claiming to be a complete run of its own.
        resume_state = run / "weights" / f"pretrain_{arm['hidden_size']}_resume.pt"
        if not resume_state.is_file():
            parser.error(f"--resume 1 but no resume state at {resume_state}")
        work = run / "work"
        work.mkdir(exist_ok=True)
    else:
        run.mkdir(parents=True, exist_ok=False)
        work = run / "work"
        work.mkdir()

    if profile == "check":
        data = run / "input.jsonl"
        write_texts(data, islice(read_texts(ROOT / "lab/data/v2/tokenizer_train.jsonl"), 16))
        batch, accumulation, seq = 2, 2, 128
        command = [sys.executable, "-u", str(ROOT / "minimind/trainer/train_pretrain.py"),
                   "--device", args.device, "--num_workers", "0", "--from_weight", "none",
                   "--from_resume", "0", "--epochs", "1", "--seed", str(args.seed),
                   "--batch_size", str(batch), "--accumulation_steps", str(accumulation),
                   "--max_seq_len", str(seq), "--learning_rate", "0.0002",
                   "--tokenizer_path", str(tokenizer), "--data_path", str(data),
                   "--save_dir", str(run / "weights"), "--save_weight", "pretrain",
                   "--log_interval", "1", "--save_interval", "1000"]
    elif profile == "full":
        data = ROOT / "lab/data/v2/train.jsonl"
        batch, accumulation, seq = 8, 4, 512
        command = [sys.executable, "-u", str(ROOT / "minimind/trainer/train_pretrain.py"),
                   "--device", args.device, "--num_workers", "0", "--from_weight", "none",
                   "--from_resume", "0", "--epochs", "1", "--seed", str(args.seed),
                   "--batch_size", str(batch), "--accumulation_steps", str(accumulation),
                   "--max_seq_len", str(seq), "--learning_rate", "0.0002",
                   "--tokenizer_path", str(tokenizer), "--data_path", str(data),
                   "--save_dir", str(run / "weights"), "--save_weight", "pretrain",
                   "--log_interval", "100", "--save_interval", "1000"]
    else:  # pack -- 决策文档 §1.5
        ledger = run / "run_ledger.jsonl"
        data = None  # pack profile reads packed memmaps, not a jsonl corpus
        command = [sys.executable, "-u", str(ROOT / "minimind/trainer/train_pretrain.py"),
                   "--device", args.device, "--num_workers", "0", "--from_weight", "none",
                   "--from_resume", "0", "--seed", str(args.seed),
                   "--batch_size", str(args.batch_size),
                   "--accumulation_steps", str(args.accumulation_steps),
                   "--max_seq_len", str(args.max_seq_len),
                   "--learning_rate", str(args.learning_rate),
                   "--tokenizer_path", str(tokenizer),
                   "--save_dir", str(run / "weights"), "--save_weight", "pretrain",
                   "--log_interval", str(args.log_interval),
                   "--packed", str(prefix), "--total_tokens", str(total_tokens),
                   "--resume", str(args.resume),
                   "--schedule", args.schedule, "--warmup_frac", str(args.warmup_frac),
                   "--decay_frac", str(args.decay_frac), "--lr_floor", str(args.lr_floor),
                   "--grad_clip", str(args.grad_clip), "--dtype", args.dtype,
                   "--keep_last_n", str(args.keep_last_n),
                   "--ledger", str(ledger), "--save_steps_frac", str(args.save_steps_frac)]
        extra_manifest["ledger"] = str(ledger)
        # The trainer only autocasts on cuda; on mps/cpu --dtype is a no-op and the
        # run is fp32 throughout. Echoing the command line alone would file a local
        # smoke run as "bf16" forever, so record what actually ran.
        extra_manifest["requested_dtype"] = args.dtype
        extra_manifest["effective_dtype"] = args.dtype if "cuda" in args.device else "float32"
    for key, value in arm.items():
        command.extend(["--" + key, str(value)])
    manifest = {"arm": args.arm, "profile": profile, "config": arm,
                "command": command, "seed": args.seed, "torch": torch.__version__,
                "data_sha256": sha256_file(data) if data is not None else None,
                "tokenizer_sha256": tokenizer_sha256,
                "initialization": "random; no model weights loaded",
                "status": "running", **extra_manifest}
    write_json(run / "manifest.json", manifest)
    started = time.perf_counter()
    with open(run / "train.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            manifest["status"] = "interrupted"
            write_json(run / "manifest.json", manifest)
            raise
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["status"] = "complete" if code == 0 else "failed"
    manifest["exit_code"] = code
    write_json(run / "manifest.json", manifest)
    sys.exit(code)
