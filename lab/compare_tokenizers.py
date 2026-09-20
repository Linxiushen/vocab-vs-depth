"""Compare complete, identical held-out documents without token truncation."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from common import read_texts, sha256_file, write_json


def compare(data, candidates, max_seq_len=512):
    texts = list(read_texts(data))
    if not texts:
        raise ValueError("Empty comparison corpus")
    rows = []
    for name, path in candidates:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        groups = {key: {"documents": 0, "chars": 0, "bytes": 0, "tokens": 0}
                  for key in ("all", "contains_cjk", "no_cjk")}
        encoded = tok(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        truncated = padding = 0
        for text, ids in zip(texts, encoded):
            if tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
                raise AssertionError(f"Lossy tokenizer {name}")
            category = "contains_cjk" if any("\u4e00" <= c <= "\u9fff" for c in text) else "no_cjk"
            for group in ("all", category):
                stats = groups[group]
                stats["documents"] += 1
                stats["chars"] += len(text)
                stats["bytes"] += len(text.encode("utf-8"))
                stats["tokens"] += len(ids)
            truncated += len(ids) > max_seq_len - 2
            padding += max(0, max_seq_len - len(ids) - 2)
        for stats in groups.values():
            stats["chars_per_token"] = stats["chars"] / stats["tokens"] if stats["tokens"] else None
            stats["bytes_per_token"] = stats["bytes"] / stats["tokens"] if stats["tokens"] else None
        row = {"name": name, "path": str(Path(path).resolve()), "vocab_size": len(tok),
               "tokenizer_sha256": sha256_file(Path(path) / "tokenizer.json"),
               "groups": groups, "roundtrip_failures": 0,
               "truncated_documents_at_seq_len": truncated,
               "padding_fraction_at_seq_len": padding / (len(texts) * max_seq_len)}
        rows.append(row)
        print(f"{name}: {groups['all']['tokens']:,} tokens; "
              f"{groups['all']['chars_per_token']:.4f} chars/token", flush=True)
    return {"data_sha256": sha256_file(data), "documents": len(texts),
            "max_seq_len": max_seq_len, "tokenizers": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", action="append", required=True, help="name=local_path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.data, [item.split("=", 1) for item in args.tokenizer])
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
