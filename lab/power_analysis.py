"""Paired bootstrap over held-out documents for a BPB difference.

BPB is a ratio of summed NLL to summed bytes, not a mean of per-document values,
so the confidence interval is obtained by resampling documents and recomputing
the ratio on each resample. Two evaluations must cover the same document set in
the same order and be produced by the same eval protocol.

This bounds MEASUREMENT noise on a fixed validation set only. It says nothing
about seed-to-seed training variance, which is a separate and usually larger
source of uncertainty; estimating that requires training each arm more than once.

Usage:
    python lab/power_analysis.py --a lab/results/random-eval-v2.json \
        --b lab/results/smoke-eval-v2.json --out lab/results/power-random-vs-smoke.json
"""
import argparse
import json
import math
import random

from common import write_json


def load(path):
    payload = json.loads(open(path, encoding="utf-8").read())
    rows = payload["per_document"]
    return payload, rows


def bpb(nll, nbytes, index):
    total_bytes = sum(nbytes[i] for i in index)
    if not total_bytes:
        raise ValueError("Resample selected zero bytes")
    return sum(nll[i] for i in index) / (math.log(2) * total_bytes)


def analyse(path_a, path_b, resamples, seed):
    meta_a, rows_a = load(path_a)
    meta_b, rows_b = load(path_b)
    if meta_a.get("protocol") != meta_b.get("protocol"):
        raise ValueError("Evaluations use different protocols; not comparable")
    if [r["index"] for r in rows_a] != [r["index"] for r in rows_b]:
        raise ValueError("Evaluations cover different documents or ordering")
    if [r["utf8_bytes"] for r in rows_a] != [r["utf8_bytes"] for r in rows_b]:
        raise ValueError("Byte counts differ; evaluations scored different text")

    nbytes = [r["utf8_bytes"] for r in rows_a]
    nll_a = [r["nll_nats"] for r in rows_a]
    nll_b = [r["nll_nats"] for r in rows_b]
    n = len(nbytes)
    whole = list(range(n))
    point_a, point_b = bpb(nll_a, nbytes, whole), bpb(nll_b, nbytes, whole)

    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        index = [rng.randrange(n) for _ in range(n)]
        diffs.append(bpb(nll_a, nbytes, index) - bpb(nll_b, nbytes, index))
    diffs.sort()
    mean = sum(diffs) / len(diffs)
    stderr = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]

    result = {
        "protocol": meta_a.get("protocol"),
        "a": {"path": path_a, "bpb": point_a, "weight_sha256": meta_a.get("weight_sha256"),
              "initialization": meta_a.get("initialization")},
        "b": {"path": path_b, "bpb": point_b, "weight_sha256": meta_b.get("weight_sha256"),
              "initialization": meta_b.get("initialization")},
        "documents": n,
        "utf8_bytes": sum(nbytes),
        "bpb_difference_a_minus_b": point_a - point_b,
        "bootstrap": {"resamples": resamples, "seed": seed, "stderr": stderr,
                      "ci95_low": lo, "ci95_high": hi},
        "minimum_detectable_difference_2se": 2 * stderr,
        "minimum_detectable_relative_to_b": 2 * stderr / point_b if point_b else None,
        "excludes_zero": lo > 0 or hi < 0,
        "covers": "measurement noise on this fixed validation set only",
        "does_not_cover": "seed-to-seed training variance; needs repeated training runs",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="baseline eval JSON")
    parser.add_argument("--b", required=True, help="comparison eval JSON")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = analyse(args.a, args.b, args.resamples, args.seed)
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
