"""Inspect a pretrained checkpoint with plain-text continuation prompts."""
import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

from common import ROOT, sha256_file, write_json

sys.path.insert(0, str(ROOT / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", default=str(ROOT / "minimind/out/smoke_512.pth"))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default=str(ROOT / "lab/results/smoke-samples.json"))
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(ROOT / "minimind/model", local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8))
    model.load_state_dict(torch.load(args.weight, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    samples = []
    for prompt in args.prompt or ["人工智能是", "学习语言模型需要", "今天的天气"]:
        ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            output = model.generate(torch.tensor([ids], device=device),
                                    max_new_tokens=args.max_new_tokens, do_sample=False,
                                    repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id)
        generated = output[0, len(ids):].tolist()
        samples.append({"prompt": prompt, "continuation": tok.decode(generated, skip_special_tokens=True),
                        "generated_tokens": len(generated)})
    result = {"weight_sha256": sha256_file(args.weight), "mode": "plain continuation, BOS prefix",
              "do_sample": False, "repetition_penalty": 1.1, "samples": samples}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
