"""Fix common text-chunk boundaries for the v3 cross-arm BPB eval protocol.

Why this exists (决策-20260918.md §1.2 D4, §2.3): the v2 eval protocol lets each
arm's own tokenizer decide where a document gets cut into <=510-token windows.
Because tokenizer A (bpe_6400) needs strictly more tokens than the nested
tokenizer B (bpe_16384) to cover the same text, A resets its BOS-anchored
context more often than B does on the very same held-out documents -- a
tokenizer-length artifact, not a fact about how well either model compresses
bytes. `--mirror-confusion` in lab/eval_bpb.py quantifies exactly how large
this artifact is on lab/data/v2/val.jsonl.

v3 removes the artifact by fixing chunk boundaries ONCE, in characters, using
only tokenizer A's <=510-token budget, and having every arm score the
identical [cs, ce) text span. This is sound only because A's merge table is a
verified byte-identical prefix of B's (see the assertions in
`assert_nested_family` below): whenever a span is <=510 A-tokens it is
provably also <=510 B-tokens (len_B(t) <= len_A(t) for every token sequence
t). We still re-check that per chunk instead of trusting the theorem blindly.

Finding the boundary itself is not a one-shot offset lookup. BPE is not a
monotone function of a prefix of its input: the token boundary that
offset_mapping reports for the full remaining text can fall in the middle of
a multi-token pretokenizer "word", and re-encoding just the truncated prefix
in isolation (which is exactly what happens at eval time, and what packing
does too) can then need MORE tokens than the same span needed inside the
longer string. So candidate cuts are found from offset_mapping and then
verified by a real, standalone re-encode of the candidate substring, backing
off one token at a time (bounded, see MAX_BACKOFF_STEPS) whenever the real
count overshoots the budget.

What v3 does NOT remove: the boundaries are A's token boundaries, so they land
mid-token for B and B pays a few extra tokens re-starting at each cut. On the
frozen lab/data/v2/val.jsonl fixture that residual is +32 B-tokens out of
413,143 (+0.008%) against 0 for A. A is not always exactly zero, because the
same BPE non-monotonicity described above can also make a split span cost one
token LESS than it did inside the whole document; either way the leftover
asymmetry is tiny and points at A. Its size depends on the --val actually
used, so no number for it is hard-coded here: every run recomputes both the
whole-document and the per-chunk token sums and writes them, with their
difference, to the manifest as `residual_boundary_token_overhead`, and
lab/eval_bpb.py copies that field into every v3 result JSON so it stays next
to the headline BPB difference instead of being an unrecorded assumption.
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from common import ROOT, read_texts, sha256_file, write_json

MAX_BACKOFF_STEPS = 128

# 决策-20260918.md §1.2 D4 names these values literally, and "A equals B" is NOT
# a substitute for them: with dropout on, BPE is randomised on both sides and the
# constructive guarantee len_B(t) <= len_A(t) -- which is what lets A's 510-token
# budget bound B's token count on the identical span -- fails even though the two
# settings agree. Same for ignore_merges (bypasses the merge table entirely, so
# the prefix relation stops implying anything) and for a pre_tokenizer that splits
# the text differently from the ByteLevel one both tokenizers were trained with.
REQUIRED_BPE_SWITCHES = (("dropout", None), ("fuse_unk", False),
                         ("byte_fallback", False), ("ignore_merges", False))
REQUIRED_PRE_TOKENIZER = (("type", "ByteLevel"), ("add_prefix_space", False),
                          ("use_regex", True))


def exactly(value, required):
    """True only for the required value itself (True/1 and False/0 differ here)."""
    if required is None or isinstance(required, bool):
        return value is required
    return value == required


def load_tokenizer_json(path):
    file = Path(path) / "tokenizer.json"
    if not file.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json under {path}")
    return json.loads(file.read_text(encoding="utf-8"))


def assert_nested_family(path_a, path_b):
    """Raise unless A's BPE is byte-identically the prefix tokenizer of B's.

    This is the precondition the whole v3 protocol depends on: without it,
    "<=510 A-tokens implies <=510 B-tokens" is not guaranteed and chunk
    boundaries picked from A alone could silently overflow B's budget.
    """
    ja, jb = load_tokenizer_json(path_a), load_tokenizer_json(path_b)
    model_a, model_b = ja["model"], jb["model"]
    if model_a.get("type") != "BPE" or model_b.get("type") != "BPE":
        raise ValueError("assert_nested_family expects BPE models on both sides")

    merges_a, merges_b = model_a["merges"], model_b["merges"]
    if len(merges_a) == 0 or len(merges_a) > len(merges_b):
        raise ValueError(
            f"Not a nested pair: A has {len(merges_a)} merges, B has {len(merges_b)}"
        )
    if merges_b[:len(merges_a)] != merges_a:
        raise ValueError(
            "Nesting broken: A's merge list is not an exact prefix of B's merge list"
        )

    vocab_a, vocab_b = model_a["vocab"], model_b["vocab"]
    mismatched = [tok for tok, tid in vocab_a.items() if vocab_b.get(tok) != tid]
    if mismatched:
        raise ValueError(
            f"Nesting broken: {len(mismatched)} of A's vocab ids disagree with B, "
            f"e.g. {mismatched[:5]!r}"
        )

    for switch, required in REQUIRED_BPE_SWITCHES:
        if model_a.get(switch) != model_b.get(switch):
            raise ValueError(
                f"BPE switch {switch!r} differs: A={model_a.get(switch)!r} "
                f"B={model_b.get(switch)!r} -- the len_B<=len_A guarantee assumes "
                f"identical dropout/fuse_unk/byte_fallback/ignore_merges settings"
            )
        for side, model in (("A", model_a), ("B", model_b)):
            if not exactly(model.get(switch), required):
                raise ValueError(
                    f"{side}'s BPE switch {switch!r} is {model.get(switch)!r}, but the v3 "
                    f"protocol requires {required!r} (决策-20260918.md §1.2 D4). Equal-but-unsafe "
                    f"settings on both sides still break len_B<=len_A: e.g. dropout makes BPE "
                    f"randomised, ignore_merges bypasses the merge table the nesting argument "
                    f"is built on. Refusing to fix chunk boundaries with this tokenizer pair."
                )

    pre_a, pre_b = ja.get("pre_tokenizer"), jb.get("pre_tokenizer")
    if pre_a != pre_b:
        raise ValueError(
            f"pre_tokenizer differs: A={pre_a!r} B={pre_b!r}"
        )
    for side, pre in (("A", pre_a), ("B", pre_b)):
        if not isinstance(pre, dict):
            raise ValueError(f"{side} has no pre_tokenizer object: {pre!r}")
        for field, required in REQUIRED_PRE_TOKENIZER:
            if not exactly(pre.get(field), required):
                raise ValueError(
                    f"{side}'s pre_tokenizer {field!r} is {pre.get(field)!r}, but the v3 "
                    f"protocol requires {required!r} -- 决策-20260918.md §1.2 D4 pins "
                    f"ByteLevel(add_prefix_space=false, use_regex=true) because a different "
                    f"pre-tokenisation changes which byte spans the merge table ever sees."
                )

    bos_a = next((t["id"] for t in ja.get("added_tokens", []) if t["content"] == "<|im_start|>"), None)
    bos_b = next((t["id"] for t in jb.get("added_tokens", []) if t["content"] == "<|im_start|>"), None)
    if bos_a is None or bos_a != bos_b:
        raise ValueError(f"BOS (<|im_start|>) id differs or missing: A={bos_a!r} B={bos_b!r}")


def find_chunk_end(text, cs, tokenizer_a, budget):
    """Return ce such that len(tokenizer_a.encode(text[cs:ce])) <= budget.

    Greedy: take the first `budget` tokens of a fresh encoding of text[cs:],
    then verify by really re-encoding that exact substring, backing off one
    token at a time (from the same offset list) until the real count fits.
    """
    remainder = text[cs:]
    encoded = tokenizer_a(remainder, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if not ids:
        raise ValueError(f"Tokenizer A produced zero tokens for a nonempty remainder at char {cs}")
    if len(ids) <= budget:
        return cs + len(remainder)  # rest of the document fits in one chunk

    k = budget
    for step in range(MAX_BACKOFF_STEPS + 1):
        if k < 1:
            raise ValueError(
                f"Chunk boundary search collapsed to zero tokens at char {cs}; "
                f"budget={budget} is too small for the pretokenizer word starting there"
            )
        ce = cs + offsets[k - 1][1]
        probe = tokenizer_a.encode(text[cs:ce], add_special_tokens=False)
        if len(probe) <= budget:
            return ce
        k -= 1
    raise ValueError(
        f"Chunk boundary search at char {cs} did not converge within "
        f"{MAX_BACKOFF_STEPS} backoff steps (budget={budget}); BPE non-monotonicity "
        f"is worse than expected for this document, investigate rather than raise the cap"
    )


def chunk_document(text, tokenizer_a, budget):
    if not text:
        raise ValueError("Cannot chunk an empty document")
    spans = []
    cs = 0
    n = len(text)
    while cs < n:
        ce = find_chunk_end(text, cs, tokenizer_a, budget)
        if ce <= cs:
            raise ValueError(f"Non-advancing chunk boundary at char {cs}")
        spans.append((cs, ce))
        cs = ce
    if spans[-1][1] != n:
        raise ValueError("Chunks do not reach the end of the document")
    joined = "".join(text[a:b] for a, b in spans)
    if joined != text:
        raise ValueError("Chunk spans do not reconstruct the original document losslessly")
    return spans


def summarize(counts):
    if not counts:
        raise ValueError("Cannot summarize an empty list of chunk token counts")
    return {"min": min(counts), "mean": sum(counts) / len(counts), "max": max(counts)}


def build(args):
    """Write eval_chunks_v3.jsonl and its sidecar manifest, or write neither.

    The chunk file goes to a temporary sibling and is moved into place only
    after every document passed every assertion and the manifest is computed.
    Streaming straight onto the published path would, on any mid-run raise,
    leave a truncated chunk file next to a manifest still describing the
    PREVIOUS complete run -- an out_sha256 pointing at content that no longer
    exists, which lab/eval_bpb.py would then happily score a subset of.
    """
    assert_nested_family(args.tokenizer_a, args.tokenizer_b)
    tok_a = AutoTokenizer.from_pretrained(args.tokenizer_a, local_files_only=True)
    tok_b = AutoTokenizer.from_pretrained(args.tokenizer_b, local_files_only=True)
    if not tok_a.is_fast or not tok_b.is_fast:
        raise ValueError("Both tokenizers must be fast (Rust-backed) for offset_mapping")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_name(out_path.stem + ".meta.json")
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")

    a_tokens_per_chunk, b_tokens_per_chunk = [], []
    total_bytes, n_chunks, n_docs = 0, 0, 0
    a_tokens_whole_doc = b_tokens_whole_doc = 0
    try:
        with open(tmp_out, "w", encoding="utf-8") as stream:
            for doc_index, text in enumerate(read_texts(args.val)):
                n_docs += 1
                # Whole-document counts are the baseline the per-chunk sums are
                # compared against below: the gap is exactly what each arm pays
                # for restarting at boundaries it did not choose.
                a_tokens_whole_doc += len(tok_a.encode(text, add_special_tokens=False))
                b_tokens_whole_doc += len(tok_b.encode(text, add_special_tokens=False))
                for chunk_index, (cs, ce) in enumerate(chunk_document(text, tok_a, args.budget)):
                    span = text[cs:ce]
                    a_ids = tok_a.encode(span, add_special_tokens=False)
                    if len(a_ids) > args.budget:
                        raise ValueError(f"doc {doc_index} chunk {chunk_index}: {len(a_ids)} A-tokens > budget {args.budget}")
                    b_ids = tok_b.encode(span, add_special_tokens=False)
                    if len(b_ids) > args.budget:
                        raise ValueError(
                            f"doc {doc_index} chunk {chunk_index}: {len(b_ids)} B-tokens > budget "
                            f"{args.budget} even though A needed only {len(a_ids)} -- nesting "
                            f"guarantee (len_B<=len_A) is violated, stop and investigate"
                        )
                    nbytes = len(span.encode("utf-8"))
                    a_tokens_per_chunk.append(len(a_ids))
                    b_tokens_per_chunk.append(len(b_ids))
                    total_bytes += nbytes
                    n_chunks += 1
                    row = {"doc": doc_index, "chunk": chunk_index, "cs": cs, "ce": ce, "utf8_bytes": nbytes}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if n_docs % 2000 == 0:
                    print(f"Chunked {n_docs} documents, {n_chunks} chunks so far", flush=True)

        if not n_docs or not n_chunks:
            raise ValueError(
                f"No evaluable documents in --val {args.val} ({n_docs} documents, {n_chunks} "
                f"chunks). An empty chunk file is never a valid artifact -- lab/eval_bpb.py "
                f"raises 'No evaluable documents' in the mirror-image case -- and the usual "
                f"cause is pointing --val at a file another line is still writing."
            )

        a_sum = sum(a_tokens_per_chunk)
        b_sum = sum(b_tokens_per_chunk)
        manifest = {
            "val": args.val, "val_sha256": sha256_file(args.val),
            "tokenizer_a": args.tokenizer_a,
            "tokenizer_a_sha256": sha256_file(Path(args.tokenizer_a) / "tokenizer.json"),
            "tokenizer_b": args.tokenizer_b,
            "tokenizer_b_sha256": sha256_file(Path(args.tokenizer_b) / "tokenizer.json"),
            "budget": args.budget,
            "documents": n_docs,
            "chunks": n_chunks,
            "total_utf8_bytes": total_bytes,
            "a_tokens_per_chunk": summarize(a_tokens_per_chunk),
            "b_tokens_per_chunk": summarize(b_tokens_per_chunk),
            "a_tokens_sum_per_chunk": a_sum,
            "a_tokens_whole_doc": a_tokens_whole_doc,
            "b_tokens_sum_per_chunk": b_sum,
            "b_tokens_whole_doc": b_tokens_whole_doc,
            "residual_boundary_token_overhead": {
                "a": a_sum - a_tokens_whole_doc,
                "b": b_sum - b_tokens_whole_doc,
                "a_pct_of_whole_doc": (a_sum - a_tokens_whole_doc) / a_tokens_whole_doc * 100,
                "b_pct_of_whole_doc": (b_sum - b_tokens_whole_doc) / b_tokens_whole_doc * 100,
                "note": "chunk boundaries are A's own token boundaries, so A's per-chunk "
                        "sum matches its whole-document count up to BPE non-monotonicity "
                        "(a token either way), while B restarts mid-token at every cut and "
                        "pays a small positive overhead; this is the residual, A-favouring "
                        "asymmetry v3 does NOT remove -- report it next to any headline "
                        "BPB difference",
            },
            "out": str(out_path),
            "out_sha256": sha256_file(tmp_out),
        }
        write_json(tmp_manifest, manifest)
        # Publish both files only once everything above succeeded, chunk file
        # first so a crash between the two moves is caught by eval_bpb.py's
        # out_sha256 check rather than passing as a consistent pair.
        os.replace(tmp_out, out_path)
        os.replace(tmp_manifest, manifest_path)
    finally:
        for leftover in (tmp_out, tmp_manifest):
            if leftover.exists():
                leftover.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True)
    parser.add_argument("--tokenizer-a", default=str(ROOT / "lab/tokenizers/bpe_6400"))
    parser.add_argument("--tokenizer-b", default=str(ROOT / "lab/tokenizers/bpe_16384"))
    parser.add_argument("--budget", type=int, default=510)
    parser.add_argument("--out", default=str(ROOT / "lab/data/v3/eval_chunks_v3.jsonl"))
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    build(args)
