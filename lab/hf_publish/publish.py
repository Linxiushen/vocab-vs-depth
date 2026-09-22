#!/usr/bin/env python3
"""Publish both vocab-vs-depth arms to the Hugging Face Hub.

Requires an authenticated session first (the token is never read or stored by
this script beyond what huggingface_hub does):

    .venv/bin/hf auth login

Then:

    .venv/bin/python lab/hf_publish/publish.py            # dry run, prints the plan
    .venv/bin/python lab/hf_publish/publish.py --execute  # create repos and upload

Each arm goes to its own repo with its own tokenizer, licensed cc-by-nc-4.0.
The card metadata deliberately carries no base_model and no pipeline_tag.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHT_NAME = "pretrain_768_step0007339.pth"

ARMS = {
    "A": {
        "repo": "vocab-vs-depth-armA-v6400-L8",
        "run": "pilot-A-s42-T120M",
        "tokenizer": "bpe_6400",
        "eval": "armA-T120M-eval-v3.json",
        "card": "README-armA.md",
    },
    "B": {
        "repo": "vocab-vs-depth-armB-v16384-L7",
        "run": "pilot-B-s42-T120M",
        "tokenizer": "bpe_16384",
        "eval": "armB-T120M-eval-v3.json",
        "card": "README-armB.md",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(arm):
    spec = ARMS[arm]
    weight = ROOT / "lab/runs" / spec["run"] / "weights" / WEIGHT_NAME
    tokenizer = ROOT / "lab/tokenizers" / spec["tokenizer"]
    card = Path(__file__).parent / spec["card"]
    evaluation = ROOT / "lab/results" / spec["eval"]
    for path in (weight, tokenizer, card, evaluation):
        if not path.exists():
            raise SystemExit(f"missing: {path}")
    expected = json.loads(evaluation.read_text(encoding="utf-8"))["weight_sha256"]
    actual = sha256(weight)
    if actual != expected:
        raise SystemExit(
            f"arm {arm}: weight sha256 {actual} does not match the evaluated "
            f"weight {expected}; refusing to publish a checkpoint that is not "
            f"the one the reported bpb was measured on"
        )
    return spec, weight, tokenizer, card, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="actually create the repos and upload; otherwise dry run")
    parser.add_argument("--arm", choices=["A", "B"], action="append",
                        help="publish only this arm (repeatable); default is both")
    args = parser.parse_args()
    arms = args.arm or ["A", "B"]

    resolved = {arm: plan(arm) for arm in arms}
    for arm in arms:
        spec, weight, tokenizer, card, digest = resolved[arm]
        print(f"arm {arm} -> Linxiushen/{spec['repo']}")
        print(f"  {weight.name}  ({weight.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]}...)")
        for item in sorted(tokenizer.iterdir()):
            print(f"  {item.name}")
        print(f"  README.md  (from {card.name})")
    if not args.execute:
        print("\ndry run; re-run with --execute to publish")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]
    print(f"\nauthenticated as {who}")
    for arm in arms:
        spec, weight, tokenizer, card, _ = resolved[arm]
        repo_id = f"{who}/{spec['repo']}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
        api.upload_file(path_or_fileobj=str(weight), path_in_repo=WEIGHT_NAME,
                        repo_id=repo_id, repo_type="model")
        for item in sorted(tokenizer.iterdir()):
            api.upload_file(path_or_fileobj=str(item), path_in_repo=item.name,
                            repo_id=repo_id, repo_type="model")
        print(f"published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    sys.exit(main())
