"""Containment-style near-duplicate audit of val against train (决策-20260918.md §1.2 D2).

Why containment, not Jaccard>=0.8: Jaccard divides shared content by the union
of BOTH documents, so it only fires when the two documents are similar in
their ENTIRETY. It cannot see a real and common leak pattern -- a short val
document (or a val paragraph) copied verbatim inside one paragraph of a much
longer train document. There, shared-windows / union-of-all-windows is tiny
even though every word of the val document leaked, so Jaccard reports "not a
duplicate" while the val score for that document is not measuring
generalization at all. Containment (shared content relative to the SMALLER
side, i.e. relative to the val document) is the right question: "how much of
this val document appears verbatim somewhere in train", not "how similar are
the two documents overall".

Why a bare window COUNT is not containment (this is the bug this file was
rewritten to fix). The first implementation flagged a val document as soon as
some train document reproduced >= --min-hits distinct 64-char window hashes.
Two things are wrong with that:

  * It is not a character measure. --min-hits distinct windows on the stride
    grid cover `window + (min_hits-1)*stride` characters at minimum, i.e. 128
    characters at (64, 32, 3), not 3*64 = 192. Both the old docstring and the
    decision doc said "≥192 字符重合"; that arithmetic double-counts the
    stride overlap.
  * It has no notion of how much of the val document that is. Measured on the
    real corpus, 3 shared windows is exactly what a shared *boilerplate
    prefix* produces: val document 14479 (1,596 chars) shared 6 windows with
    train line 1,232,274 -- all of them at offsets 0/32/64/96/128/160, inside
    a 227-character "You are an AI assistant. ..." preamble the two documents
    have in common, while `val[300:400] in train_text` is False. The other 86%
    of the document never appears in train. That is not a leak, it is a
    template. The damage is not random: of the 304 documents the old rule
    removed, 52.30% were >80% ASCII (English) against 8.58% in the documents
    it kept, and 23.36% started with that one preamble against 0.06% kept. The
    whole point of this project is a BPB comparison between a 6,400 and a
    16,384 token vocabulary, and the two vocabularies do NOT compress Latin
    script and Chinese alike -- so a val filter that deletes English documents
    6x more often than Chinese ones moves the dependent variable directly.

What this file does instead, per (val document, train document) pair:

  1. Template filter. During a first streaming pass over train we count, for
     every val window hash, in how many distinct train documents it occurs
     (its document frequency). A hash occurring in more than --max-doc-freq
     train documents (default 100 out of 1.25M) is boilerplate -- shared
     licence headers, chat preambles, formatting scaffolding -- and carries no
     evidence about this particular document having leaked. It is dropped.
  2. Coverage, not counts. During a second pass we take the matched windows'
     character spans in the VAL document, merge the overlapping intervals, and
     divide by len(val_text). A pair is flagged when, after the template
     filter, the merged coverage is >= --min-coverage (default 0.5) AND the
     number of distinct matched non-template hashes is >= min(--min-hits,
     number of windows this val document has).
  3. Full-containment escape hatch. Independently, a pair is flagged when the
     UNfiltered merged coverage is >= --full-containment (default 0.98): if
     essentially the whole val document is reproduced verbatim inside one
     train document, it is leaked whether or not its sentences are also common
     elsewhere. This branch is also what makes short documents detectable at
     all -- see below.

The `min(--min-hits, n_windows)` floor fixes a second real hole in the old
rule: a document shorter than 2*window has 1 or 2 windows in total, so at
--min-hits 3 it could never be flagged no matter how completely it appeared in
train. 9.46% of val_large (1,864 documents, shortest 11 chars) sat in that
blind spot. The report now publishes that population under
`val_documents_below_min_hits_windows` instead of leaving it implicit.

Method details: slide a --window (default 64) character window with --stride
(32) over every val document; hash each window with blake2b(digest_size=8)
into `hash -> [(val_doc_id, start, end), ...]`. Stream train twice through
common.read_texts (a lazy generator -- the 1.2GB corpus is never materialized
as a list) and window it the same way: pass 1 counts document frequency, pass
2 scores pairs. Two passes cost about twice one pass (~17s here) and are the
only way to know a hash is boilerplate before deciding with it.

Coverage note: windows are taken at 0, stride, 2*stride, ... while a full
window still fits, PLUS one extra window flush against the end of the text if
the stride grid doesn't already land there -- otherwise a leak sitting in the
last <window> characters of a document could be missed depending on its
length mod stride. Documents shorter than --window use the whole document as
a single window.

blake2b at digest_size=8 can collide in principle (64-bit digests over ~3e5
val windows). A collision can only manufacture a false positive, never hide a
real leak, and the coverage rule now needs half a document's worth of
colliding windows rather than three, so the residual risk is far below the
level at which it would be worth a second verification pass over the text.

Usage:
    python lab/neardup_audit.py --val lab/data/v3/val_large.jsonl \
        --train lab/data/v3/train.jsonl --window 64 --stride 32 --min-hits 3 \
        --min-coverage 0.5 --max-doc-freq 100 \
        --out lab/data/v3/neardup_report.json [--apply] [--limit N]

--apply rewrites val_large.jsonl and val_manifest.jsonl (dropping hit
documents, keeping remaining order, reindexing "i") and APPENDS a record to
the sibling manifest.json's near_duplicate_check log with the method,
parameters and number of documents removed. It refuses to run against a
--limit'd (partial) train scan, and refuses to touch files whose sha256 does
not match what that manifest.json says it froze. Every output is written to a
temporary file and os.replace'd into position, so a reader either sees the
whole old file or the whole new one, and a failed assertion never leaves a
half-written validation set behind.
"""
import argparse
from hashlib import blake2b
import json
import os
from pathlib import Path
import time

from common import ROOT, read_texts, sha256_file, text_key, write_json

SENSITIVITY_MIN_HITS = (1, 2, 3, 4, 5)
SENSITIVITY_COVERAGE = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def window_spans(text, size, stride):
    """Yield (start, end, blake2b8) for each window; see "Coverage note" above."""
    n = len(text)
    if n <= size:
        yield 0, n, blake2b(text.encode("utf-8"), digest_size=8).digest()
        return
    start = 0
    last_start = None
    while start + size <= n:
        yield start, start + size, \
            blake2b(text[start:start + size].encode("utf-8"), digest_size=8).digest()
        last_start = start
        start += stride
    tail_start = n - size
    if tail_start != last_start:
        yield tail_start, n, \
            blake2b(text[tail_start:n].encode("utf-8"), digest_size=8).digest()


def window_hash_set(text, size, stride):
    return {h for _, _, h in window_spans(text, size, stride)}


def merged_span_length(spans):
    """Characters covered by a set of possibly overlapping [start, end) spans."""
    spans = sorted(spans)
    total = 0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    return total + cur_end - cur_start


def build_val_table(val_texts, size, stride):
    table = {}
    window_counts = []
    for doc_id, text in enumerate(val_texts):
        count = 0
        for start, end, h in window_spans(text, size, stride):
            table.setdefault(h, []).append((doc_id, start, end))
            count += 1
        window_counts.append(count)
    return table, window_counts


def scan_document_frequency(train_path, table, size, stride, limit):
    """Pass 1: how many distinct train documents contain each val window hash."""
    freq = {}
    lines = 0
    for line_no, text in enumerate(read_texts(train_path)):
        if limit and line_no >= limit:
            break
        lines += 1
        for h in window_hash_set(text, size, stride):
            if h in table:
                freq[h] = freq.get(h, 0) + 1
    return freq, lines


def frequency_histogram(freq):
    buckets = {"1": 0, "2": 0, "3-10": 0, "11-100": 0, "101-1000": 0, ">1000": 0}
    for count in freq.values():
        if count == 1:
            buckets["1"] += 1
        elif count == 2:
            buckets["2"] += 1
        elif count <= 10:
            buckets["3-10"] += 1
        elif count <= 100:
            buckets["11-100"] += 1
        elif count <= 1000:
            buckets["101-1000"] += 1
        else:
            buckets[">1000"] += 1
    return buckets


def scan_pairs(args, table, template, val_lens, val_window_counts, limit):
    """Pass 2: score every (train document, val document) pair that shares a window.

    Returns hit_records (val_doc_id -> {"train_hits": [...], "n_train_hits": int}),
    plus the per-val-document maxima the sensitivity tables are computed from.
    """
    size, stride = args.window, args.stride
    hit_records = {}
    best_raw_windows = {}       # legacy criterion: max distinct shared hashes
    best_raw_coverage = {}      # unfiltered merged coverage
    best_filtered_cov = {}      # doc_id -> [cov at >=1 window, ..., cov at >=5]
    lines = 0
    for line_no, text in enumerate(read_texts(args.train)):
        if limit and line_no >= limit:
            break
        lines += 1
        raw_hashes = {}         # doc_id -> set(hash)
        raw_spans = {}          # doc_id -> [(start, end)]
        flt_hashes = {}
        flt_spans = {}
        for h in window_hash_set(text, size, stride):
            entries = table.get(h)
            if not entries:
                continue
            is_template = h in template
            for doc_id, start, end in entries:
                raw_hashes.setdefault(doc_id, set()).add(h)
                raw_spans.setdefault(doc_id, []).append((start, end))
                if not is_template:
                    flt_hashes.setdefault(doc_id, set()).add(h)
                    flt_spans.setdefault(doc_id, []).append((start, end))

        for doc_id, hashes in raw_hashes.items():
            n_raw = len(hashes)
            if n_raw > best_raw_windows.get(doc_id, 0):
                best_raw_windows[doc_id] = n_raw

            val_len = val_lens[doc_id]
            spans = flt_spans.get(doc_id)
            n_flt = len(flt_hashes.get(doc_id, ()))
            flt_cov = 0.0
            if spans:
                flt_cov = merged_span_length(spans) / val_len
                slot = best_filtered_cov.setdefault(doc_id, [0.0] * (len(SENSITIVITY_MIN_HITS) + 1))
                for m in SENSITIVITY_MIN_HITS:
                    if n_flt >= m and flt_cov > slot[m]:
                        slot[m] = flt_cov

            # Unfiltered coverage is only needed for the full-containment
            # branch; skip the interval merge when it cannot possibly reach
            # the threshold (total matched window length is an upper bound).
            raw_cov = 0.0
            if len(raw_spans[doc_id]) * size >= args.full_containment * val_len:
                raw_cov = merged_span_length(raw_spans[doc_id]) / val_len
                if raw_cov > best_raw_coverage.get(doc_id, 0.0):
                    best_raw_coverage[doc_id] = raw_cov

            required = min(args.min_hits, val_window_counts[doc_id])
            by_coverage = n_flt >= required and flt_cov >= args.min_coverage
            by_containment = raw_cov >= args.full_containment
            if not (by_coverage or by_containment):
                continue
            record = hit_records.setdefault(doc_id, {"train_hits": [], "n_train_hits": 0})
            record["n_train_hits"] += 1
            if len(record["train_hits"]) < args.max_hits_per_doc:
                # For a pair we are actually reporting, compute the unfiltered
                # coverage for real rather than leaving the 0.0 the cheap
                # upper-bound shortcut above would otherwise print.
                raw_cov = merged_span_length(raw_spans[doc_id]) / val_len
                record["train_hits"].append({
                    "train_line": line_no,
                    "shared_windows": n_flt,
                    "shared_windows_unfiltered": n_raw,
                    "covered_chars": round(flt_cov * val_len),
                    "coverage": round(flt_cov, 6),
                    "coverage_unfiltered": round(raw_cov, 6),
                    "rule": "coverage" if by_coverage else "full_containment",
                })
    return hit_records, best_raw_windows, best_raw_coverage, best_filtered_cov, lines


def sensitivity_tables(best_raw_windows, best_filtered_cov, n_val):
    legacy = {}
    for m in SENSITIVITY_MIN_HITS:
        flagged = sum(1 for v in best_raw_windows.values() if v >= m)
        legacy[str(m)] = {"hit_val_documents": flagged, "hit_rate": flagged / n_val}
    coverage = {}
    for m in SENSITIVITY_MIN_HITS:
        row = {}
        for c in SENSITIVITY_COVERAGE:
            flagged = sum(1 for slot in best_filtered_cov.values() if slot[m] >= c)
            row[f"{c:.2f}"] = flagged
        coverage[str(m)] = row
    return legacy, coverage


def audit(args):
    val_texts = list(read_texts(args.val))
    if not val_texts:
        raise ValueError(f"{args.val}: no documents to audit")
    val_hashes = [text_key(t) for t in val_texts]
    val_lens = [len(t) for t in val_texts]
    table, val_window_counts = build_val_table(val_texts, args.window, args.stride)

    started = time.perf_counter()
    limit = args.limit
    freq, lines_pass1 = scan_document_frequency(args.train, table, args.window,
                                                args.stride, limit)
    template = {h for h, count in freq.items() if count > args.max_doc_freq}
    hit_records, best_raw_windows, best_raw_coverage, best_filtered_cov, lines_pass2 = \
        scan_pairs(args, table, template, val_lens, val_window_counts, limit)
    if lines_pass1 != lines_pass2:
        raise AssertionError(f"train line count changed between passes: "
                             f"{lines_pass1} then {lines_pass2}; the file moved under us")
    elapsed = time.perf_counter() - started

    hit_doc_ids = sorted(hit_records)
    legacy_table, coverage_table = sensitivity_tables(best_raw_windows, best_filtered_cov,
                                                      len(val_texts))
    below = sum(1 for c in val_window_counts if c < args.min_hits)
    limited = bool(limit)
    report = {
        "method": "containment_window_coverage",
        "criterion": (f"flag a val document when some train document covers >= "
                      f"{args.min_coverage} of its characters with non-template "
                      f"windows (>= min({args.min_hits}, n_windows) distinct hashes), "
                      f"or reproduces >= {args.full_containment} of it verbatim"),
        "reason_not_jaccard": "Jaccard>=0.8 misses containment leaks (a short val "
                              "document copied into one paragraph of a long train "
                              "document has tiny Jaccard but total containment)",
        "reason_not_window_count": "a bare shared-window count fires on shared "
                                   "boilerplate prefixes and removed English documents "
                                   "6x more often than Chinese ones, biasing the very "
                                   "BPB the study measures",
        "hash": "blake2b digest_size=8",
        "window": args.window,
        "stride": args.stride,
        "min_hits": args.min_hits,
        "min_coverage": args.min_coverage,
        "full_containment": args.full_containment,
        "max_doc_freq": args.max_doc_freq,
        "min_hits_covers_characters": args.window + (args.min_hits - 1) * args.stride,
        "val_documents_total": len(val_texts),
        "val_documents_below_min_hits_windows": below,
        "val_windows_total": sum(val_window_counts),
        "val_window_hashes_distinct": len(table),
        "template_hashes_filtered": len(template),
        "document_frequency_histogram": frequency_histogram(freq),
        "val_path": str(args.val),
        "val_sha256": sha256_file(args.val),
        "train_path": str(args.train),
        "train_sha256": sha256_file(args.train),
        "train_lines_scanned": lines_pass2,
        "limited": limited,
        "limit": limit if limited else None,
        "elapsed_seconds": elapsed,
        "lines_per_second": (2 * lines_pass2) / elapsed if elapsed > 0 else None,
        "hit_val_documents": len(hit_doc_ids),
        "hit_rate": len(hit_doc_ids) / len(val_texts),
        "sensitivity_legacy_window_count_only": legacy_table,
        "sensitivity_coverage_by_min_hits": coverage_table,
        "full_containment_val_documents": sum(
            1 for v in best_raw_coverage.values() if v >= args.full_containment),
        "max_hits_per_doc_recorded": args.max_hits_per_doc,
        "hits": [
            {"val_doc": doc_id, "val_sha256": val_hashes[doc_id],
             "val_chars": val_lens[doc_id],
             "train_documents_matched": hit_records[doc_id]["n_train_hits"],
             "train_hits": hit_records[doc_id]["train_hits"]}
            for doc_id in hit_doc_ids
        ],
    }
    if args.out:
        write_json(args.out, report)
    return report, val_texts, val_hashes, hit_doc_ids


def _atomic_write_lines(path, lines):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for line in lines:
            stream.write(line)
    os.replace(tmp, path)


def apply_removal(args, report, val_texts, val_hashes, hit_doc_ids):
    if report["limited"]:
        raise ValueError("--apply requires a full (unlimited) scan of train; "
                         "a --limit'd run can only undercount near-duplicates")
    val_path = Path(args.val)
    val_manifest_path = Path(args.val_manifest) if args.val_manifest else \
        val_path.with_name("val_manifest.jsonl")
    manifest_path = Path(args.manifest) if args.manifest else \
        val_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"--apply needs an existing manifest.json at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Never rewrite a file this manifest did not freeze: --apply is destructive
    # and in-place, and pointing it at the wrong --val used to succeed silently
    # while stamping "audit performed" onto an unrelated manifest.
    recorded = manifest.get("files", {})
    for label, path, key in ((" --val", val_path, "val_large.jsonl"),
                             ("--train", Path(args.train), "train.jsonl")):
        expected = recorded.get(key)
        if expected is None:
            raise ValueError(f"{manifest_path} has no files.{key} entry; refusing to "
                             f"apply removals against an unverifiable manifest")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} {path} has sha256 {actual}, but {manifest_path} "
                             f"froze files.{key} = {expected}. Refusing to rewrite a "
                             "validation set this manifest does not describe.")

    hit_set = set(hit_doc_ids)
    kept = [(i, t) for i, t in enumerate(val_texts) if i not in hit_set]
    if len(kept) + len(hit_doc_ids) != len(val_texts):
        raise AssertionError("removal accounting does not add up")
    if not kept:
        raise AssertionError("every val document was flagged; refusing to write an "
                             "empty validation set")

    prefix_len = manifest.get("historical_prefix_len", manifest.get("old_val_rows"))
    if prefix_len is None:
        raise ValueError(f"{manifest_path} has neither historical_prefix_len nor "
                         "old_val_rows; cannot keep the 30M-comparable prefix honest")
    surviving_prefix = sum(1 for i, _ in kept if i < prefix_len)
    removed_prefix = prefix_len - surviving_prefix
    # Removal preserves order, so the surviving historical documents are still
    # exactly rows [0, surviving_prefix) of the rewritten file.
    if [i for i, _ in kept[:surviving_prefix]] != [i for i, _ in kept if i < prefix_len]:
        raise AssertionError("surviving historical documents are no longer a prefix")

    new_hashes = [val_hashes[i] for i, _ in kept]
    if len(set(new_hashes)) != len(new_hashes):
        raise AssertionError("val_large has duplicate content after removal")

    _atomic_write_lines(val_path,
                        (json.dumps({"text": t}, ensure_ascii=False) + "\n"
                         for _, t in kept))
    _atomic_write_lines(val_manifest_path,
                        (json.dumps({"i": i, "sha256": h}, ensure_ascii=False) + "\n"
                         for i, h in enumerate(new_hashes)))

    record = {
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": report["method"],
        "criterion": report["criterion"],
        "window": args.window,
        "stride": args.stride,
        "min_hits": args.min_hits,
        "min_coverage": args.min_coverage,
        "full_containment": args.full_containment,
        "max_doc_freq": args.max_doc_freq,
        "hash": "blake2b digest_size=8",
        "train_lines_scanned": report["train_lines_scanned"],
        "val_rows_before": len(val_texts),
        "val_rows_after": len(kept),
        "hit_val_documents_removed": len(hit_doc_ids),
        "historical_prefix_removed": removed_prefix,
        "hit_rate": report["hit_rate"],
        "report": str(args.out) if args.out else None,
    }
    # Append-only: an audit record is a disclosure obligation (决策 §五.9), so a
    # later run that finds nothing must not be able to overwrite the run that
    # removed 304 documents with "removed: 0".
    log = manifest.get("near_duplicate_check")
    if not isinstance(log, list):
        log = [] if log in (None, "not performed") else [log]
    log.append(record)
    manifest["near_duplicate_check"] = log
    manifest["near_duplicate_removed_cumulative"] = \
        sum(entry.get("hit_val_documents_removed", 0) for entry in log)

    manifest.setdefault("old_val_rows_before", manifest.get("old_val_rows"))
    manifest["old_val_removed"] = manifest["old_val_rows_before"] - surviving_prefix
    manifest["old_val_rows"] = surviving_prefix
    manifest["historical_prefix_len"] = surviving_prefix
    manifest["new_from_train_rows"] = len(kept) - surviving_prefix
    manifest["val_large_rows"] = len(kept)
    if manifest["old_val_rows"] + manifest["new_from_train_rows"] != manifest["val_large_rows"]:
        raise AssertionError("manifest row accounting is self-contradictory after removal")
    # The held-out set is frozen the moment removals land: make_val_large.py
    # refuses to rebuild over a frozen directory, so re-running step 1 of the
    # runbook can no longer silently undo this audit.
    manifest["frozen"] = True
    manifest["frozen_reason"] = ("near-duplicate audit applied; rebuilding this "
                                 "directory would discard the audit and change "
                                 "val_large.jsonl under downstream eval artifacts")
    manifest.setdefault("files", {})
    manifest["files"]["val_large.jsonl"] = sha256_file(val_path)
    manifest["files"]["val_manifest.jsonl"] = sha256_file(val_manifest_path)
    write_json(manifest_path, manifest)
    return {"removed": len(hit_doc_ids), "remaining": len(kept),
            "historical_prefix_len": surviving_prefix,
            "historical_prefix_removed": removed_prefix,
            "val_path": str(val_path), "val_manifest_path": str(val_manifest_path),
            "manifest_path": str(manifest_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="fraction of the val document that must be covered by "
                             "non-template matched windows (default 0.5)")
    parser.add_argument("--full-containment", type=float, default=0.98,
                        help="unfiltered coverage at which a pair is flagged regardless "
                             "of the template filter (default 0.98)")
    parser.add_argument("--max-doc-freq", type=int, default=100,
                        help="a window hash occurring in more than this many train "
                             "documents is boilerplate and is not counted as evidence")
    parser.add_argument("--max-hits-per-doc", type=int, default=20,
                        help="cap on train hits listed per val document in the report; "
                             "the full count is kept in train_documents_matched")
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--val-manifest", help="default: val_manifest.jsonl next to --val")
    parser.add_argument("--manifest", help="default: manifest.json next to --val")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan only the first N train lines (0 = all); for timing runs")
    args = parser.parse_args()
    if args.window < args.stride:
        parser.error("--window must be >= --stride, or windows would skip characters")
    if args.window < 1 or args.stride < 1 or args.min_hits < 1:
        parser.error("--window, --stride and --min-hits must all be >= 1 "
                     "(--stride 0 would never advance the window)")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not 0.0 < args.min_coverage <= 1.0:
        parser.error("--min-coverage must be in (0, 1]")
    if not 0.0 < args.full_containment <= 1.0:
        parser.error("--full-containment must be in (0, 1]")
    if args.max_doc_freq < 1:
        parser.error("--max-doc-freq must be >= 1")
    if args.max_hits_per_doc < 1:
        parser.error("--max-hits-per-doc must be >= 1")
    # Checked BEFORE the scan: --apply used to be rejected only after the report
    # had already been written, so a refused command still replaced the
    # authoritative full-scan report with a 0-hit partial one.
    if args.apply and args.limit:
        parser.error("--apply requires a full scan; drop --limit (a partial scan can "
                     "only undercount near-duplicates)")
    if args.limit:
        out_path = Path(args.out)
        if out_path.is_file():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                existing = None
            if isinstance(existing, dict) and existing.get("limited") is False:
                parser.error(f"{out_path} holds a full-scan report ("
                             f"{existing.get('train_lines_scanned')} lines); refusing to "
                             "overwrite it with a --limit'd partial scan. Use a different "
                             "--out for timing runs.")

    report, val_texts, val_hashes, hit_doc_ids = audit(args)
    print(json.dumps({k: v for k, v in report.items() if k != "hits"},
                     ensure_ascii=False, indent=2))
    print(f"{len(report['hits'])} val documents with hits (list omitted above; see --out)")

    if args.apply:
        outcome = apply_removal(args, report, val_texts, val_hashes, hit_doc_ids)
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
