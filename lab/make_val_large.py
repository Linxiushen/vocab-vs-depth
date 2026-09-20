"""Freeze the v3 held-out set: grow val from 1,943 to 20,000 documents while
keeping every original v2 document first, in its original order, so 30M-scale
history stays comparable document-for-document (see 决策-20260918.md §1.2 D1).

Selection is a pure content-hash sort, not a random sample: for every v2
train document we compute sha256(text) and take the hex-ascending smallest
(target - len(old_val)) *unique* content hashes that are not already in old
val. Hex order carries no meaning of its own (it is not a "hard" or "typical"
ordering) -- it is used only because it is a single deterministic total order
that needs no RNG stream, no seed, and no external library to reproduce: a
third party with the same train.jsonl gets the same 18,057 documents.

Two things this script guards against that the one-line description in the
decision doc does not spell out, because real data has them:
  * `lab/data/v2/train.jsonl` contains 190 exact-duplicate lines (verified by
    a full-file scan: 1,268,294 lines / 1,268,104 unique sha256(text)). If a
    duplicated hash were selected, keeping only its first occurrence in val
    would leave the remaining copies sitting in train with the same content
    -> nonzero train/val overlap. So every line whose hash is selected is
    removed from train in its entirety, and val gets exactly one copy. This is
    why train.jsonl ends up at 1,250,235 rows and not the 1,250,237 the
    decision doc's arithmetic predicts: 2 of the 18,057 selected hashes happen
    to have a duplicate line in train.
  * If an old-val document's content ever turned out to also appear in
    train.jsonl (measured to be false today: 0 of 1,943), those train lines
    are stripped too, so "exact_content_overlap_train_val == 0" is an
    assertion checked against the files written to disk, not a property
    assumed from the algorithm.

Overwrite protection. This script is step 1 of the runbook and
`lab/neardup_audit.py --apply` is step 2; step 2 removes documents from
val_large.jsonl in place. Re-running step 1 on a directory step 2 has already
touched would silently restore the removed documents, reset
near_duplicate_check to "not performed", and leave every downstream artifact
pinned to a val_sha256 that no longer exists. So: an output directory that
already holds a manifest.json needs --force, and one whose manifest says
frozen:true (set by --apply) additionally needs --unfreeze.

Every output is written to a temporary file and os.replace'd into position.
train.jsonl is 1.2GB and other lines of work stream it while this runs; an
atomic swap means a concurrent reader sees either the whole old file or the
whole new one, never a truncated prefix.

Usage:
    python lab/make_val_large.py --train lab/data/v2/train.jsonl \
        --old-val lab/data/v2/val.jsonl --target 20000 --out lab/data/v3
"""
import argparse
import datetime
import json
import os
from pathlib import Path

from common import ROOT, read_texts, sha256_file, text_key, write_json


def _tmp_path(path):
    return path.with_name(path.name + ".tmp")


def write_jsonl_rows(path, rows):
    path = Path(path)
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_texts_counting(path, texts, commit=True):
    """Like common.write_texts, but atomic and returns how many lines it wrote.

    With commit=False the content is left in the sibling .tmp file and the
    caller does the os.replace once it has validated the row counts, so a run
    that fails its accounting assertions leaves the previous file in place
    instead of a truncated or empty replacement.
    """
    path = Path(path)
    tmp = _tmp_path(path)
    n = 0
    with open(tmp, "w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    if commit:
        os.replace(tmp, path)
    return n


def load_old_val(path):
    texts = list(read_texts(path))
    hashes = [text_key(t) for t in texts]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{path}: old val file has internal duplicate content, "
                          "refusing to silently drop rows from the historical baseline")
    return texts, hashes


def scan_train_hashes(path):
    """Pass 1: sha256(text) for every train line, in order, duplicates kept."""
    return [text_key(t) for t in read_texts(path)]


def split_train(path, exclude_set, selected_set, out_train):
    """Pass 2: stream train once, capture selected texts, write the remainder.

    A generator (not a list) feeds write_texts_counting so the whole 1.2GB
    file is never held in memory at once -- only the up-to-`len(selected_set)`
    captured texts and the small exclude/selected hash sets are.

    `dropped` counts the LINES removed (not the hashes), which is what the
    row-accounting assertion needs: duplicate lines mean the two differ.
    """
    captured = {}
    dropped = [0]

    def remainder():
        for text in read_texts(path):
            h = text_key(text)
            if h in exclude_set:
                dropped[0] += 1
                if h in selected_set and h not in captured:
                    captured[h] = text
                continue
            yield text

    kept = write_texts_counting(out_train, remainder(), commit=False)
    return captured, kept, dropped[0]


def check_output_dir(out_dir, force, unfreeze):
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        existing = {}
    if existing.get("frozen") and not unfreeze:
        raise ValueError(
            f"{manifest_path} says frozen:true ({existing.get('frozen_reason')}). "
            "Rebuilding would restore documents the near-duplicate audit removed and "
            "invalidate every artifact pinned to val_large.jsonl sha256 "
            f"{existing.get('files', {}).get('val_large.jsonl')}. Pass --force --unfreeze "
            "only if you intend to redo step 2 of the runbook afterwards.")
    if not force:
        raise ValueError(
            f"{manifest_path} already exists (val_large_rows="
            f"{existing.get('val_large_rows')}); refusing to overwrite a held-out set "
            "that downstream artifacts may already be pinned to. Pass --force, or point "
            "--out at a new directory.")


def upstream_provenance(train_path):
    """Carry the v2 manifest's upstream source fingerprint into v3, if present.

    scripts/rebuild_val.py needs to know which upstream revision val_large.jsonl
    was derived from. Reading it from the sibling manifest of --train keeps a
    single source of truth instead of a constant hard-coded in two files, and
    costs nothing: the upstream file itself is not re-hashed here.
    """
    sibling = Path(train_path).with_name("manifest.json")
    if not sibling.is_file():
        return None, None
    try:
        recorded = json.loads(sibling.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    return recorded.get("source"), recorded.get("source_sha256")


def build(args):
    train_path = Path(args.train)
    old_val_path = Path(args.old_val)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_output_dir(out_dir, args.force, args.unfreeze)

    old_texts, old_hashes = load_old_val(old_val_path)
    old_hash_set = set(old_hashes)
    n_old = len(old_texts)
    n_new = args.target - n_old
    if n_new <= 0:
        raise ValueError(f"--target {args.target} must exceed old val size {n_old}")

    train_hashes = scan_train_hashes(train_path)
    train_lines = len(train_hashes)
    unique_train_hashes = set(train_hashes)
    duplicate_lines_in_train = train_lines - len(unique_train_hashes)

    candidates = sorted(unique_train_hashes - old_hash_set)
    if len(candidates) < n_new:
        raise ValueError(f"only {len(candidates)} unique train documents available, "
                          f"need {n_new} to reach --target {args.target}")
    selected = candidates[:n_new]
    selected_set = set(selected)

    # Anything whose content already sits in val (old or newly selected) must
    # not remain in train, even if it wasn't picked by the hex-sort itself
    # (e.g. an old-val document accidentally duplicated inside train.jsonl).
    exclude_set = selected_set | old_hash_set
    old_hashes_found_in_train = len(old_hash_set & unique_train_hashes)

    out_train = out_dir / "train.jsonl"
    captured, train_rows_after, dropped_lines = split_train(
        train_path, exclude_set, selected_set, out_train)

    # Row accounting, checked while the new train.jsonl is still only a .tmp.
    # Without these two, "overlap == 0" and "no internal duplicates" both hold
    # trivially on an empty train.jsonl, so the assertion set could not tell a
    # correct run from one that deleted the corpus.
    def _abort(message):
        tmp = _tmp_path(out_train)
        if tmp.is_file():
            os.unlink(tmp)
        raise AssertionError(message)

    if len(captured) != n_new:
        _abort(f"expected to capture {n_new} new val documents, got {len(captured)}")
    if train_rows_after <= 0:
        _abort("train.jsonl would be empty after the split; refusing to freeze a corpus "
               "with no training data")
    if train_rows_after != train_lines - dropped_lines:
        _abort(f"train row accounting: {train_lines} read - {dropped_lines} dropped != "
               f"{train_rows_after} written (did --train change between the two passes?)")
    os.replace(_tmp_path(out_train), out_train)

    new_texts = [captured[h] for h in selected]
    val_large_texts = old_texts + new_texts
    if len(val_large_texts) != args.target:
        raise AssertionError(f"val_large has {len(val_large_texts)} rows, expected {args.target}")

    out_val_large = out_dir / "val_large.jsonl"
    write_texts_counting(out_val_large, val_large_texts)

    # --- hard assertions, re-checked against the files actually written ---
    written_val_hashes = [text_key(t) for t in read_texts(out_val_large)]
    if written_val_hashes[:n_old] != old_hashes:
        raise AssertionError("val_large does not start with the original val, in original order")
    if len(set(written_val_hashes)) != len(written_val_hashes):
        raise AssertionError("val_large contains duplicate content")
    if n_old + n_new != args.target:
        raise AssertionError(f"{n_old} + {n_new} != {args.target}")

    written_val_hash_set = set(written_val_hashes)
    overlap = 0
    for text in read_texts(out_train):
        if text_key(text) in written_val_hash_set:
            overlap += 1
    if overlap != 0:
        raise AssertionError(f"train and val_large share {overlap} documents by exact content")

    val_manifest_path = out_dir / "val_manifest.jsonl"
    write_jsonl_rows(val_manifest_path,
                      ({"i": i, "sha256": h} for i, h in enumerate(written_val_hashes)))

    upstream_source, upstream_sha = upstream_provenance(train_path)
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": (f"sha256(text) hex-ascending sort over {train_path}; "
                   "take the smallest (target - old_val_rows) unique content hashes "
                   f"not already present in {old_val_path}; union with old val kept "
                   "first, in its original order; all lines whose content hash is "
                   "selected (all occurrences, in case of duplicate lines) are removed "
                   "from train"),
        "inputs": {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "old_val_path": str(old_val_path),
            "old_val_sha256": sha256_file(old_val_path),
            "upstream_source": upstream_source,
            "upstream_source_sha256": upstream_sha,
        },
        "seed": args.seed,
        "seed_note": ("recorded for manifest-schema parity with lab/data/v2/manifest.json; "
                      "selection itself is a deterministic content-hash sort and does not "
                      "consume this value -- there is no RNG in this script"),
        "target": args.target,
        "old_val_rows": n_old,
        "new_from_train_rows": n_new,
        "historical_prefix_len": n_old,
        "historical_prefix_note": ("val_large.jsonl rows [0, historical_prefix_len) are the "
                                   "surviving v2 val documents, in v2 order. Slice with this "
                                   "field, never with a hard-coded 1943: neardup_audit.py "
                                   "--apply may remove some of them and updates it."),
        "val_large_rows": len(val_large_texts),
        "train_rows_before": train_lines,
        "train_rows_after": train_rows_after,
        "train_rows_dropped": dropped_lines,
        "train_duplicate_lines_observed": duplicate_lines_in_train,
        "old_val_hashes_found_inside_train_and_stripped": old_hashes_found_in_train,
        "exact_content_overlap_train_val": overlap,
        "val_large_internal_duplicates": len(written_val_hashes) - len(written_val_hash_set),
        "near_duplicate_check": "not performed",
        "frozen": False,
        "files": {
            "val_large.jsonl": sha256_file(out_val_large),
            "train.jsonl": sha256_file(out_train),
            "val_manifest.jsonl": sha256_file(val_manifest_path),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=str(ROOT / "lab/data/v2/train.jsonl"))
    parser.add_argument("--old-val", default=str(ROOT / "lab/data/v2/val.jsonl"))
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--out", default=str(ROOT / "lab/data/v3"))
    parser.add_argument("--seed", type=int, default=42,
                        help="unused by the algorithm; kept only for manifest-schema parity")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an --out directory that already holds a manifest.json")
    parser.add_argument("--unfreeze", action="store_true",
                        help="additionally required when that manifest says frozen:true, "
                             "i.e. when neardup_audit.py --apply has already run")
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be >= 1")
    if args.unfreeze and not args.force:
        parser.error("--unfreeze only makes sense together with --force")
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
