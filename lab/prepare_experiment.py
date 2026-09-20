"""Freeze a content-disjoint holdout and a global reservoir sample for BPE."""
import argparse
import json
from pathlib import Path
import random

from common import read_texts, sha256_file, text_key, write_json, write_texts


def prepare(source, holdout, exclude, out, sample_size=300_000, seed=42):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    excluded = {text_key(text) for text in read_texts(exclude)}
    candidates = list(read_texts(holdout))
    overlap = sum(text_key(text) in excluded for text in candidates)
    validation = {text_key(text): text for text in candidates if text_key(text) not in excluded}
    if not validation or sample_size <= 0:
        raise ValueError("Need a nonempty holdout and a positive sample size")

    rng = random.Random(seed)
    sample = []
    train_rows = heldout_rows = 0
    seen_val = set()
    with open(out / "train.jsonl", "w", encoding="utf-8") as stream:
        for text in read_texts(source):
            key = text_key(text)
            if key in validation:
                heldout_rows += 1
                seen_val.add(key)
                continue
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            train_rows += 1
            if len(sample) < sample_size:
                sample.append(text)
            else:
                index = rng.randrange(train_rows)
                if index < sample_size:
                    sample[index] = text
    if seen_val != set(validation):
        raise ValueError("Holdout contains documents absent from source")
    val_texts = list(validation.values())
    random.Random(seed).shuffle(val_texts)
    rng.shuffle(sample)
    write_texts(out / "val.jsonl", val_texts)
    write_texts(out / "tokenizer_train.jsonl", sample)
    if {text_key(t) for t in sample} & set(validation):
        raise AssertionError("Tokenizer sample overlaps validation")
    report = {
        "seed": seed,
        "source": str(Path(source).resolve()),
        "source_sha256": sha256_file(source),
        "previous_holdout_sha256": sha256_file(holdout),
        "smoke_train_sha256": sha256_file(exclude),
        "previous_holdout_rows": len(candidates),
        "previous_holdout_rows_seen_by_smoke": overlap,
        "train_rows": train_rows,
        "heldout_source_rows": heldout_rows,
        "val_unique_documents": len(val_texts),
        "tokenizer_sample_rows": len(sample),
        "tokenizer_sample_unique_documents": len({text_key(t) for t in sample}),
        "exact_content_overlap_train_val": 0,
        "exact_content_overlap_smoke_val": 0,
        "near_duplicate_check": "not performed",
        "files": {name: sha256_file(out / name) for name in
                  ("train.jsonl", "val.jsonl", "tokenizer_train.jsonl")},
    }
    write_json(out / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.holdout, args.exclude, args.out,
                             args.sample_size, args.seed), indent=2), flush=True)
