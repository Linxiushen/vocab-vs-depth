"""Recover measurable facts about the completed smoke run without ETA guesses."""
import json
import re
import sys

import torch
from transformers import AutoTokenizer

from common import ROOT, read_texts, sha256_file, write_json

sys.path.insert(0, str(ROOT / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "minimind/model", local_files_only=True)
    texts = list(read_texts(ROOT / "minimind/dataset/smoke_pretrain.jsonl"))
    full_body_tokens = supervised_body_tokens = truncated = 0
    padding_positions = 0
    for start in range(0, len(texts), 1000):
        encoded = tokenizer(texts[start:start + 1000], add_special_tokens=False,
                            return_attention_mask=False)["input_ids"]
        for ids in encoded:
            full_body_tokens += len(ids)
            body = ids[:510]
            supervised_body_tokens += sum(token != tokenizer.pad_token_id for token in body)
            padding_positions += max(0, 512 - len(body) - 2)
            truncated += len(ids) > 510
    weights = ROOT / "minimind/out/smoke_512.pth"
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    resume = torch.load(ROOT / "minimind/checkpoints/smoke_512_resume.pth",
                        map_location="cpu", weights_only=True)
    log = (ROOT / "lab/results/smoke-20260910.log").read_text()
    losses = re.findall(r"\((\d+)/5000\), loss: ([0-9.]+)", log)
    optimizer_steps = {int(state["step"].item()) for state in resume["optimizer"]["state"].values()}
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "documents": len(texts), "sequence_length": 512,
        "checkpoint_step": resume["step"], "epoch_index": resume["epoch"],
        "micro_batches": len(texts) // 8, "optimizer_steps": sorted(optimizer_steps),
        "processed_positions_including_padding": len(texts) * 512,
        "full_body_tokens_before_truncation": full_body_tokens,
        "supervised_body_tokens": supervised_body_tokens,
        "supervised_targets_including_eos": supervised_body_tokens + len(texts),
        "padding_positions": padding_positions,
        "padding_fraction": padding_positions / (len(texts) * 512),
        "truncated_documents": truncated,
        "last_logged_microbatch_loss": float(losses[-1][1]),
        "last_20_logged_microbatch_mean_loss": sum(float(x[1]) for x in losses[-20:]) / 20,
        "weight_bytes": weights.stat().st_size, "weight_sha256": sha256_file(weights),
        "log_sha256": sha256_file(ROOT / "lab/results/smoke-20260910.log"),
        "timing_note": "Historical /tmp log creation to final write was about 119 minutes; active compute time was not recorded.",
    }
    write_json(ROOT / "lab/results/smoke-audit.json", result)
    print(json.dumps(result, indent=2))
