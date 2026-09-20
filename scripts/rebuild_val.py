"""Reconstruct val_large.jsonl from the public upstream corpus, for anyone who
does NOT have (and is not entitled to redistribute) our processed text.

Why this script exists: lab/data/v3/val_large.jsonl is derived from
jingyaogong/minimind_dataset, which is released under a non-commercial
license (see DATA_LICENSE.md). We do not redistribute that derived text.
What we do publish is lab/data/v3/val_manifest.jsonl -- a small,
non-substitutable list of {"i": position, "sha256": sha256(text)} rows -- plus
this script. Anyone who separately downloads the ORIGINAL upstream file
minimind/dataset/pretrain_t2t_mini.jsonl can look each hash up in that file
and reassemble byte-identical val_large.jsonl themselves, without us ever
having shipped the text. A content hash is not a "license workaround": it lets
a downstream user who is independently entitled to hold the upstream file
verify and use exactly the same evaluation set we did, while we ship 0 bytes
of NC-licensed text.

This only works because the "text" field is carried verbatim from the
upstream file through every processing step in this project (split, filter,
union) -- no normalization, truncation, or re-encoding ever touches it. If
that ever stops being true, this script's hash lookup silently reconstructs
the wrong document for the affected id, and the only thing that catches it is
the whole-file sha256 comparison against the recorded manifest.json. That
check is therefore not optional, and this file no longer treats it as
optional: a --manifest that is given but missing is an error, and the one way
to skip verification (--manifest '') prints a warning to stderr and exits
non-zero so an unverified rebuild can never be mistaken for a verified one.

Nothing is written to --out until the rebuilt bytes have been verified. The
reconstruction goes to a sibling .tmp file and is os.replace'd into position
only after the sha256 matches, because the earlier version of this script
wrote --out first and checked afterwards -- with --out defaulting to the
frozen lab/data/v3/val_large.jsonl, a verification failure destroyed the very
file it was verifying. --out now defaults to a .rebuilt.jsonl sidecar, and
overwriting an existing --out requires --force.

Usage:
    python scripts/rebuild_val.py \
        --upstream minimind/dataset/pretrain_t2t_mini.jsonl \
        --val-manifest lab/data/v3/val_manifest.jsonl \
        --manifest lab/data/v3/manifest.json \
        --out lab/data/v3/val_large.rebuilt.jsonl
    diff lab/data/v3/val_large.jsonl lab/data/v3/val_large.rebuilt.jsonl
"""
import argparse
import json
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "lab"))
from common import ROOT, read_texts, sha256_file, text_key  # noqa: E402

# Fallback only. The authoritative value is inputs.upstream_source_sha256 in the
# manifest.json that froze the validation set; keeping a constant here as well
# lets --manifest '' still check something, but the manifest always wins so the
# two cannot drift apart unnoticed.
FALLBACK_UPSTREAM_SHA256 = "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"

EXIT_UNVERIFIED = 3


def load_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(
            f"--manifest {manifest_path} does not exist. The rebuilt file can only be "
            "trusted if its sha256 is compared against the manifest that froze the "
            "validation set; a mistyped path is a user error, not a request to skip "
            "verification. Pass --manifest '' if you really have no manifest.")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def load_targets(val_manifest_path):
    targets = []
    for line_number, line in enumerate(open(val_manifest_path, encoding="utf-8")):
        row = json.loads(line)
        if row.get("i") != line_number:
            raise ValueError(f"{val_manifest_path}:{line_number+1}: expected \"i\"=="
                              f"{line_number}, got {row.get('i')} (manifest must be "
                              "sequential and in output order)")
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"{val_manifest_path}:{line_number+1}: missing/malformed sha256")
        targets.append(sha)
    if not targets:
        raise ValueError(f"{val_manifest_path}: no rows")
    return targets


def collect_from_upstream(upstream_path, wanted):
    """Single streaming pass: capture the first occurrence of each wanted hash.

    Stops early once every wanted hash has been captured -- with the target
    set near the head of a large upstream file this can be much faster than a
    full scan, and it is always at least as fast, never slower.
    """
    remaining = set(wanted)
    captured = {}
    for text in read_texts(upstream_path):
        if not remaining:
            break
        h = text_key(text)
        if h in remaining:
            captured[h] = text
            remaining.discard(h)
    return captured, remaining


def rebuild(args):
    manifest, manifest_path = (None, None)
    if args.manifest:
        manifest, manifest_path = load_manifest(args.manifest)

    expected_upstream = FALLBACK_UPSTREAM_SHA256
    expected_source = None
    if manifest:
        recorded = manifest.get("inputs", {}).get("upstream_source_sha256")
        expected_source = manifest.get("inputs", {}).get("upstream_source")
        if recorded:
            expected_upstream = recorded

    upstream_sha = sha256_file(args.upstream)
    if not args.skip_upstream_hash_check and upstream_sha != expected_upstream:
        raise ValueError(
            f"{args.upstream} has sha256 {upstream_sha}, expected {expected_upstream}"
            + (f" (recorded in {manifest_path} as {expected_source})" if expected_source else "")
            + ". This script reconstructs val_large.jsonl by exact-content lookup against "
            "a pinned upstream file; a different upstream revision will silently fail to "
            "find some hashes or, worse, find the wrong document if minimind_dataset is "
            "ever re-released with edited text. Pass --skip-upstream-hash-check only if "
            "you have verified the mismatch is benign.")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise ValueError(f"{out_path} already exists; pass --force to replace it, or point "
                         "--out somewhere else. (Rebuilding on top of a frozen "
                         "val_large.jsonl is exactly the accident this guard exists for.)")

    targets = load_targets(args.val_manifest)
    captured, missing = collect_from_upstream(args.upstream, targets)
    if missing:
        raise ValueError(f"{len(missing)} of {len(targets)} target documents were not "
                          f"found in {args.upstream}; upstream file does not match the "
                          "corpus val_large.jsonl was built from")

    texts = [captured[h] for h in targets]
    for i, (text, expected_hash) in enumerate(zip(texts, targets)):
        if text_key(text) != expected_hash:
            raise AssertionError(f"row {i}: reconstructed text hash does not match "
                                  "its own lookup key (should be impossible)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    rebuilt_sha = sha256_file(tmp_path)

    result = {
        "upstream": str(args.upstream),
        "upstream_sha256": upstream_sha,
        "val_manifest": str(args.val_manifest),
        "documents_reconstructed": len(texts),
        "out": str(out_path),
        "rebuilt_sha256": rebuilt_sha,
        "verified_against_manifest": False,
    }
    if manifest is not None:
        expected = manifest.get("files", {}).get("val_large.jsonl")
        if expected is None:
            os.unlink(tmp_path)
            raise ValueError(f"{manifest_path} has no files.val_large.jsonl entry")
        if rebuilt_sha != expected:
            os.unlink(tmp_path)
            raise AssertionError(
                f"rebuilt val_large.jsonl sha256 {rebuilt_sha} != recorded {expected} in "
                f"{manifest_path}. Reconstruction does not match the frozen validation "
                f"set; nothing was written to {out_path}.")
        result["verified_against_manifest"] = True
        result["manifest_recorded_sha256"] = expected
        recorded_rows = manifest.get("val_large_rows")
        if recorded_rows is not None and recorded_rows != len(texts):
            os.unlink(tmp_path)
            raise AssertionError(f"{manifest_path} records val_large_rows={recorded_rows} "
                                 f"but {args.val_manifest} lists {len(texts)} documents")
    os.replace(tmp_path, out_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream",
                        default=str(ROOT / "minimind/dataset/pretrain_t2t_mini.jsonl"))
    parser.add_argument("--val-manifest", default=str(ROOT / "lab/data/v3/val_manifest.jsonl"))
    parser.add_argument("--manifest", default=str(ROOT / "lab/data/v3/manifest.json"),
                        help="verifies the rebuilt file's sha256 and supplies the expected "
                             "upstream sha256; pass an empty string to skip verification "
                             f"(which then exits {EXIT_UNVERIFIED})")
    parser.add_argument("--out", default=str(ROOT / "lab/data/v3/val_large.rebuilt.jsonl"))
    parser.add_argument("--force", action="store_true",
                        help="allow --out to be overwritten if it already exists")
    parser.add_argument("--skip-upstream-hash-check", action="store_true")
    args = parser.parse_args()
    if args.manifest == "":
        args.manifest = None
    outcome = rebuild(args)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    if not outcome["verified_against_manifest"]:
        print("WARNING: this rebuild was NOT verified against a manifest.json. The hash "
              "lookup cannot detect that it reassembled the wrong corpus. Do not use "
              f"{outcome['out']} as a validation set.", file=sys.stderr)
        sys.exit(EXIT_UNVERIFIED)
