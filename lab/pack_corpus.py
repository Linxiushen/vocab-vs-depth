"""Offline packing for the vocab-vs-depth experiment (decision doc S1.2 D3, S1.3).

Each arm gets its own packed stream: for every document, in a shared cross-arm
random order (CRN), emit [BOS] + tok(text) + [EOS], concatenate across documents,
and slice into fixed seq_len blocks with no padding. The tail remainder (< one
block) is dropped. Sequence boundaries fall wherever they fall in the token
stream; BOS/EOS mark real document boundaries only, never inserted at a cut.

Why two arms must share doc_order: "each arm packs independently" (decision
doc S1.1) is the point of this design (both arms end at 0% padding), but the
comparison is only fair if both arms see the same documents in the same order
(common random numbers). doc_order.i32 is therefore written once, together
with doc_order.meta.json, and the second arm's run must load and verify both.
Verifying only the permutation's sha256 would be too weak: the permutation is
a function of (seed, n_docs) alone, so two *different* corpora with the same
line count would "pass" CRN while the two arms silently trained on different
text. doc_order.meta.json pins seed, n_docs, seq_len and the sha256 of the
train file itself, and all four must match.

Why bytes are tracked per sequence, not estimated: BPB needs exact UTF-8 byte
counts of the source text consumed by each packed sequence. Token-count-based
estimates would bias the comparison in exactly the dimension being measured
(A and B tokenize the same text to different lengths). The decision doc
sketches this as `return_offsets_mapping` + `text[:end].encode('utf-8')`, but
those offsets are *character* offsets: under a ByteLevel BPE a multi-byte
character can be split across two tokens, so a cut inside a character rounds
the byte count up to the whole character (measured: 223/7278 rows off by 1-2
bytes on arm A, 55/6161 on arm B -- small, but asymmetric between arms, and
the contract says "exact"). We instead take the exact source-byte length of
every vocabulary entry from the ByteLevel alphabet (byte_length_table) and
accumulate it per token. That is byte-exact by construction, cheaper (no
per-row re-encode of a text prefix), and self-validating: every document
hard-asserts sum(byte_len[token]) == len(text.encode('utf-8')).

The four hard assertions of decision doc S1.2 D3 all crash, none warn:
  1. decode() of the packed prefix == the doc_order-ordered source text,
     byte-exact. Decoding keeps special tokens (skip_special_tokens=False) so
     that (a) a literal "<|endoftext|>" in running text does not make a
     *correct* pack fail, and (b) a spurious EOS injected at a sequence
     boundary -- the headline failure mode of decision doc S5 risk #3 -- is
     actually caught; with skip_special_tokens=True this assertion is blind
     to it. Only whole documents are compared, so the window never ends in
     the middle of a character.
  2. ids.max() < vocab_size and padding count == 0.
  3. total_tokens within +-1% of A ~302M / B ~255M. Defined only for a
     full-corpus run (no --max-docs), where it is a hard crash; measured
     A = 302,828,544 (+0.27%) and B = 256,128,512 (+0.44%) on the real
     lab/data/v3/train.jsonl, so the band is comfortable. With --max-docs
     the corpus is a prefix slice and the constants do not apply, so the
     assertion is skipped rather than rescaled: lab/data/*/train.jsonl is
     not uniformly shuffled by file position (average bytes/doc across ten
     consecutive 130k-line buckets swings from 386 to 2281), so a prorated
     bound fires on correct output. What replaces it on subset runs is a
     per-document cross-check that is stronger anyway: every document is
     tokenized twice, once in natural file order while sizing the memmap and
     once through the seek-based doc_order pass that writes the array, and
     the two token counts must match per document (this also catches a
     seek/permutation bug that preserves the aggregate total but reads the
     wrong text for an index).
  4. bytes_cum monotonic non-decreasing, final value <= corpus bytes.

Crash-safety: the ids array is built in a .tmp file and os.replace()d into
place, and packed_{arm}.meta.json is removed up front and written last, so an
interrupted run can never leave behind a full-size half-zero ids.u16 next to a
stale meta.json that lab/packed_dataset.py would happily load.
"""
import argparse
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from transformers import AutoTokenizer

from common import ROOT, sha256_file, write_json

# --arm only names the output files; the vocabulary comes from whatever
# --tokenizer is handed in. Without this table `--arm A --tokenizer bpe_16384`
# silently writes a packed_A.* built from B's vocabulary and every assertion
# still passes (assertion 2 compares against the tokenizer that was loaded).
# Digests are of the frozen lab/tokenizers/{bpe_6400,bpe_16384}/tokenizer.json.
ARM_TOKENIZER = {
    "A": ("cdd7e5bc16aa5081f00fbcff810c27de5b4971b0cba7bda970c873e31491cc9e", 6400),
    "B": ("ff99beb3a8ea206db443a3247229b637c16ba6d2faa0caab71f36dc925addfb8", 16384),
}
# Hard assertion 3, decision doc S1.2 D3 / S2.1. Only defined for a full-corpus
# run; see the module docstring for why it is not prorated onto subset runs.
FULL_CORPUS_TOKENS = {"A": 302_000_000, "B": 255_000_000}
FULL_CORPUS_TOL = 0.01


def byte_length_table(tok):
    """Exact source-UTF-8 byte length of every vocabulary entry.

    Regular pieces are spelled in GPT-2's ByteLevel alphabet, so inverting that
    alphabet gives the exact bytes a piece stands for. Added tokens are not
    byte-level; if one of them shows up inside a document's encoding (a literal
    "<|endoftext|>" in running text is the known case) it consumed exactly its
    own literal spelling from the source, so that is what it costs here.
    """
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet, extra = list(printable), 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            alphabet.append(256 + extra)
            extra += 1
    unicode_to_byte = {chr(code): value for value, code in zip(printable, alphabet)}
    added = tok.added_tokens_decoder
    table = np.zeros(len(tok), dtype=np.int64)
    for token_id, piece in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
        if token_id in added:
            table[token_id] = len(str(added[token_id].content).encode("utf-8"))
            continue
        try:
            table[token_id] = len(bytes(unicode_to_byte[ch] for ch in piece))
        except KeyError:
            raise ValueError(
                f"token id {token_id} ({piece!r}) is not spelled in the ByteLevel "
                f"alphabet; exact byte accounting is impossible for this tokenizer")
    return table


def scan_corpus(path, tok, limit, batch_size=2000):
    """One sequential pass: line byte offsets, per-doc utf8 length, per-doc
    token count (only used to size the memmap and as the other half of hard
    assertion 3's per-document cross-check)."""
    offsets, utf8_lens, token_lens = [], [], []
    batch_texts = []

    def flush():
        if not batch_texts:
            return
        enc = tok(batch_texts, add_special_tokens=False)
        token_lens.extend(len(ids) for ids in enc["input_ids"])
        batch_texts.clear()

    with open(path, "rb") as stream:
        n = 0
        while limit is None or n < limit:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            row = json.loads(raw.decode("utf-8"))
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: line {n + 1}: expected nonempty text")
            offsets.append(offset)
            utf8_lens.append(len(text.encode("utf-8")))
            batch_texts.append(text)
            n += 1
            if len(batch_texts) >= batch_size:
                flush()
                if n % 200000 == 0:
                    print(f"  scan: {n} documents", flush=True)
        flush()
    if not offsets:
        raise ValueError(f"{path}: no documents found")
    return (np.array(offsets, dtype=np.int64), np.array(utf8_lens, dtype=np.int64),
            np.array(token_lens, dtype=np.int64))


def load_or_create_doc_order(out_dir, seed, n_docs, seq_len, train_path, train_sha256):
    """Create doc_order.i32 (+ its sidecar meta) or verify the existing pair.

    The permutation only depends on (seed, n_docs), so its sha256 alone cannot
    tell two same-length corpora apart -- the sidecar pins the train file's
    sha256 and seq_len as well, which is what CRN actually requires.
    """
    path, meta_path = out_dir / "doc_order.i32", out_dir / "doc_order.meta.json"
    order = np.random.default_rng(seed).permutation(n_docs).astype(np.int32)
    order_sha256 = hashlib.sha256(order.tobytes()).hexdigest()
    meta = {
        "seed": seed, "n_docs": n_docs, "seq_len": seq_len,
        "train_sha256": train_sha256, "train_path": str(train_path),
        "doc_order_sha256": order_sha256, "numpy_version": np.__version__,
    }
    if path.exists() != meta_path.exists():
        raise ValueError(
            f"{out_dir}: exactly one of doc_order.i32 / doc_order.meta.json exists; "
            f"refusing to guess. Delete both (and the packed_* files that were "
            f"built from them) and re-pack both arms.")
    if not path.exists():
        order.tofile(path)
        write_json(meta_path, meta)
        return order, order_sha256, True
    seen = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(path)
    if actual_sha != order_sha256:
        raise ValueError(
            f"{path} sha256 {actual_sha} != {order_sha256} expected for seed={seed} "
            f"n_docs={n_docs}. Refusing to silently regenerate a different "
            f"permutation: both arms must consume the same document order (CRN).")
    for key in ("seed", "n_docs", "seq_len", "train_sha256"):
        if seen.get(key) != meta[key]:
            raise ValueError(
                f"{meta_path}: {key}={seen.get(key)!r} was used for the arm packed "
                f"earlier (train_path={seen.get('train_path')!r}), but this run has "
                f"{key}={meta[key]!r} (train_path={str(train_path)!r}). Both arms "
                f"must pack the same file with the same seed/seq_len (CRN).")
    if seen.get("numpy_version") != meta["numpy_version"]:
        raise ValueError(
            f"{meta_path}: doc_order.i32 was generated by numpy "
            f"{seen.get('numpy_version')}, this run has numpy {np.__version__}. "
            f"np.random.Generator streams are not guaranteed stable across numpy "
            f"versions (NEP 19), so the permutation may silently differ. Pin numpy "
            f"(lab/requirements-lock.txt) or copy doc_order.i32 across machines.")
    return order, order_sha256, False


def read_line_text(handle, offset):
    handle.seek(offset)
    row = json.loads(handle.readline().decode("utf-8"))
    return row["text"]


def longest_clean_prefix(text, byte_cum, n_tokens):
    """(prefix, tokens) for the first `n_tokens` text tokens of `text`.

    Only used when the packed stream is cut in the middle of a document (i.e.
    the very last document before the dropped tail): a ByteLevel cut can land
    inside a character, so back off whole tokens until the byte prefix is a
    valid UTF-8 string that decode() can also produce.
    """
    raw = text.encode("utf-8")
    while n_tokens > 0:
        try:
            return raw[:int(byte_cum[n_tokens - 1])].decode("utf-8"), n_tokens
        except UnicodeDecodeError:
            n_tokens -= 1
    return "", 0


def pack(args):
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    vocab_size = len(tok)
    if vocab_size > 65536:
        raise ValueError(f"vocab_size {vocab_size} does not fit uint16")
    bos_id, eos_id = tok.bos_token_id, tok.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer is missing bos_token_id/eos_token_id")
    tokenizer_sha256 = sha256_file(Path(args.tokenizer) / "tokenizer.json")
    expected_sha, expected_vocab = ARM_TOKENIZER[args.arm]
    if (tokenizer_sha256, vocab_size) != (expected_sha, expected_vocab):
        raise ValueError(
            f"arm {args.arm} is pinned to tokenizer sha256={expected_sha} "
            f"(vocab_size={expected_vocab}), but --tokenizer {args.tokenizer} has "
            f"sha256={tokenizer_sha256} (vocab_size={vocab_size}). --arm names the "
            f"output files, so a swap here would write packed_{args.arm}.* from the "
            f"other arm's vocabulary with every assertion still passing.")
    byte_len = byte_length_table(tok)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train)
    train_sha256 = sha256_file(train_path)

    print(f"[{args.arm}] scanning {train_path} (max_docs={args.max_docs or 'all'})", flush=True)
    doc_offsets, utf8_lens, token_lens = scan_corpus(train_path, tok, args.max_docs)
    n_docs = len(doc_offsets)
    text_tokens_all = int(token_lens.sum())
    total_utf8_bytes = int(utf8_lens.sum())
    raw_stream_tokens = text_tokens_all + 2 * n_docs  # +BOS +EOS per doc
    n_seq, dropped_tail_tokens = divmod(raw_stream_tokens, args.seq_len)
    capacity = n_seq * args.seq_len
    if n_seq < 1:
        raise ValueError("corpus too small to fill even one sequence")

    doc_order, doc_order_sha256, created = load_or_create_doc_order(
        out_dir, args.seed, n_docs, args.seq_len, train_path, train_sha256)
    print(f"[{args.arm}] doc_order.i32 "
          f"{'created' if created else 'reused (seed/n_docs/seq_len/train_sha256 verified)'} "
          f"sha256={doc_order_sha256[:16]}... train_sha256={train_sha256[:16]}...", flush=True)

    ids_path = out_dir / f"packed_{args.arm}.ids.u16"
    tmp_path = out_dir / f"packed_{args.arm}.ids.u16.tmp"
    bytes_path = out_dir / f"packed_{args.arm}.bytes_cum.i64"
    meta_path = out_dir / f"packed_{args.arm}.meta.json"
    # meta.json is the completeness sentinel for lab/packed_dataset.py: drop it
    # first, write it last, and never leave a partially filled ids.u16 in place.
    meta_path.unlink(missing_ok=True)
    try:
        ids_mm = np.memmap(tmp_path, mode="w+", dtype=np.uint16, shape=(n_seq, args.seq_len))
        flat = ids_mm.reshape(-1)
        bytes_cum = np.zeros(n_seq, dtype=np.int64)

        bos_piece, eos_piece = tok.convert_ids_to_tokens(bos_id), tok.convert_ids_to_tokens(eos_id)
        check_target = min(1000, n_seq) * args.seq_len
        check_pieces, check_tokens, check_done = [], 0, False

        stream_pos = 0
        bytes_before_doc = 0
        structural_written = 0
        endoftext_literal_hits = 0

        with open(train_path, "rb") as handle:
            batch_size = 2000
            k = 0
            while k < n_docs and stream_pos < capacity:
                batch_docs = doc_order[k:k + batch_size]
                batch_texts = [read_line_text(handle, doc_offsets[d]) for d in batch_docs]
                enc = tok(batch_texts, add_special_tokens=False)["input_ids"]
                # One lookup + one cumsum per batch, not per document: everything the
                # per-document loop below needs is then O(1) integer arithmetic into
                # batch_byte_cum (302M tokens go through here, so this matters).
                lens = np.fromiter((len(ids) for ids in enc), dtype=np.int64, count=len(enc))
                ends = np.cumsum(lens)
                starts = ends - lens
                batch_tokens = np.fromiter(chain.from_iterable(enc), dtype=np.int64,
                                           count=int(ends[-1]))
                batch_byte_cum = np.cumsum(byte_len[batch_tokens])
                endoftext_literal_hits += int(np.count_nonzero(batch_tokens == 0))
                scan_lens = token_lens[batch_docs]
                if not np.array_equal(lens, scan_lens):
                    bad = int(np.flatnonzero(lens != scan_lens)[0])
                    raise ValueError(
                        f"hard assertion 3 failed: doc {int(batch_docs[bad])} tokenized to "
                        f"{int(lens[bad])} tokens via seek+doc_order but {int(scan_lens[bad])} "
                        f"via the natural-order scan -- seek/permutation bug (misattributed "
                        f"or corrupted read)")
                if int(lens.min()) < 1:
                    raise ValueError(f"doc {int(batch_docs[int(lens.argmin())])} tokenized to "
                                     f"0 tokens")
                doc_bytes = batch_byte_cum[ends - 1] - np.where(starts > 0,
                                                               batch_byte_cum[starts - 1], 0)
                if not np.array_equal(doc_bytes, utf8_lens[batch_docs]):
                    bad = int(np.flatnonzero(doc_bytes != utf8_lens[batch_docs])[0])
                    raise ValueError(
                        f"byte accounting failed on doc {int(batch_docs[bad])}: its tokens "
                        f"account for {int(doc_bytes[bad])} source bytes, the text is "
                        f"{int(utf8_lens[batch_docs[bad]])} bytes -- tokenization is not "
                        f"byte-lossless here and bytes_cum would be wrong")

                for j in range(len(batch_docs)):
                    if stream_pos >= capacity:
                        break
                    o, l_d = int(starts[j]), int(lens[j])
                    base = int(batch_byte_cum[o - 1]) if o > 0 else 0
                    ld2 = l_d + 2
                    take = min(ld2, capacity - stream_pos)
                    flat[stream_pos] = bos_id
                    text_written = min(take - 1, l_d)
                    if text_written:
                        flat[stream_pos + 1:stream_pos + 1 + text_written] = \
                            batch_tokens[o:o + text_written]
                    if take == ld2:
                        flat[stream_pos + ld2 - 1] = eos_id
                    structural_written += 1 + (1 if take == ld2 else 0)  # BOS, then EOS

                    row_start = stream_pos // args.seq_len
                    row_end = (stream_pos + take - 1) // args.seq_len
                    for s in range(row_start, row_end + 1):
                        consumed = (s + 1) * args.seq_len - stream_pos  # doc-local entries
                        text_taken = min(max(consumed - 1, 0), l_d)  # BOS advances no byte
                        bytes_cum[s] = bytes_before_doc + (
                            int(batch_byte_cum[o + text_taken - 1]) - base if text_taken else 0)

                    if not check_done and check_tokens == stream_pos:
                        if take == ld2:
                            check_pieces.append(bos_piece + batch_texts[j] + eos_piece)
                            check_tokens += ld2
                            check_done = check_tokens >= check_target
                        else:  # stream cut inside this document: last document touched
                            prefix, used = longest_clean_prefix(
                                batch_texts[j], batch_byte_cum[o:o + l_d] - base, take - 1)
                            check_pieces.append(bos_piece + prefix)
                            check_tokens += 1 + used
                            check_done = True

                    stream_pos += take
                    if take != ld2:
                        break  # tail truncated: this is the last document touched
                    bytes_before_doc += int(doc_bytes[j])
                k += len(batch_docs)
                if k % 200000 < batch_size:
                    print(f"[{args.arm}] packed {k}/{n_docs} docs, {stream_pos}/{capacity} tokens",
                          flush=True)

        ids_mm.flush()
        del ids_mm, flat
        ids_ro = np.memmap(tmp_path, mode="r", dtype=np.uint16, shape=(n_seq, args.seq_len))

        # ---- hard assertion 1: byte-exact round trip over whole documents ----
        reference_text = "".join(check_pieces)
        decoded = tok.decode(ids_ro.reshape(-1)[:check_tokens].tolist(), skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
        if decoded != reference_text:
            first_diff = next((i for i in range(min(len(decoded), len(reference_text)))
                               if decoded[i] != reference_text[i]),
                              min(len(decoded), len(reference_text)))
            raise ValueError(
                f"hard assertion 1 failed: decode of the first {check_tokens} packed tokens "
                f"diverges from the doc_order-ordered source text at character {first_diff} "
                f"(decoded len={len(decoded)}, reference len={len(reference_text)}); "
                f"decoded={decoded[first_diff:first_diff + 40]!r} "
                f"reference={reference_text[first_diff:first_diff + 40]!r}")
        print(f"[{args.arm}] assertion 1 OK: first {check_tokens} tokens "
              f"({check_tokens / args.seq_len:.1f} sequences, {len(check_pieces)} documents, "
              f"BOS/EOS included) round-trip byte-exact "
              f"({len(reference_text.encode('utf-8'))} bytes)", flush=True)

        # ---- hard assertion 2: vocab range + zero padding + literal-<|endoftext|> report ----
        max_id = int(ids_ro.max())
        if max_id >= vocab_size:
            raise ValueError(f"hard assertion 2 failed: ids.max()={max_id} >= vocab_size={vocab_size}")
        if stream_pos != capacity:
            raise ValueError(
                f"hard assertion 2 failed: wrote {stream_pos} tokens, expected exactly "
                f"{capacity} (implies unfilled/padded cells)")
        print(f"[{args.arm}] assertion 2 OK: ids.max()={max_id} < vocab_size={vocab_size}, "
              f"0 padding cells, literal '<|endoftext|>' occurrences={endoftext_literal_hits}",
              flush=True)

        # ---- hard assertion 3: full-corpus token total (+-1%), else per-doc cross-check ----
        text_tokens = capacity - structural_written
        full_corpus_token_check = None
        if args.max_docs is None:
            reference = FULL_CORPUS_TOKENS[args.arm]
            rel_diff = (capacity - reference) / reference
            full_corpus_token_check = {
                "reference_tokens": reference, "total_tokens": capacity,
                "relative_diff": rel_diff, "tolerance": FULL_CORPUS_TOL,
            }
            if abs(rel_diff) > FULL_CORPUS_TOL:
                raise ValueError(
                    f"hard assertion 3 failed: total_tokens={capacity:,} is {rel_diff:+.2%} from "
                    f"the decision-doc figure for arm {args.arm} ({reference:,}), outside "
                    f"+-{FULL_CORPUS_TOL:.0%}. Either --train is not the full "
                    f"lab/data/v3/train.jsonl or the packing is wrong.")
            print(f"[{args.arm}] assertion 3 OK: total_tokens={capacity:,} is {rel_diff:+.2%} from "
                  f"the decision-doc reference {reference:,} (within +-{FULL_CORPUS_TOL:.0%})",
                  flush=True)
        else:
            print(f"[{args.arm}] assertion 3: full-corpus +-1% band not defined for a "
                  f"--max-docs={args.max_docs} prefix slice; the per-document token-count "
                  f"cross-check (natural-order scan vs seek+doc_order pass) covered all "
                  f"{n_docs} documents with 0 mismatches", flush=True)

        # ---- hard assertion 4: bytes_cum monotonic, bounded by corpus size ----
        if bool(np.any(np.diff(bytes_cum) < 0)):
            raise ValueError("hard assertion 4 failed: bytes_cum is not monotonically non-decreasing")
        consumed_utf8_bytes = int(bytes_cum[-1])
        if consumed_utf8_bytes > total_utf8_bytes:
            raise ValueError(
                f"hard assertion 4 failed: bytes_cum[-1]={consumed_utf8_bytes} > "
                f"total_utf8_bytes={total_utf8_bytes}")
        print(f"[{args.arm}] assertion 4 OK: bytes_cum monotonic, final={consumed_utf8_bytes} <= "
              f"total_utf8_bytes={total_utf8_bytes}", flush=True)

        del ids_ro
        ids_sha256 = sha256_file(tmp_path)
        if args.expect_ids_sha256 is not None and ids_sha256 != args.expect_ids_sha256:
            raise ValueError(
                f"--expect-ids-sha256 {args.expect_ids_sha256} != actual {ids_sha256}: this "
                f"pack is not bit-identical to the one it was supposed to reproduce "
                f"(decision doc S2.2: re-pack on the training machine and assert equality)")
        os.replace(tmp_path, ids_path)
    except BaseException:
        # never leave a half-written array behind: meta.json is already gone,
        # and the staged .tmp would only waste disk (605 MB on a full arm).
        ids_mm = flat = ids_ro = None
        tmp_path.unlink(missing_ok=True)
        raise
    bytes_cum.tofile(bytes_path)
    elapsed = time.perf_counter() - started
    meta = {
        "arm": args.arm, "tokenizer_sha256": tokenizer_sha256, "vocab_size": vocab_size,
        "n_seq": n_seq, "seq_len": args.seq_len, "bos_id": bos_id, "eos_id": eos_id,
        "total_tokens": capacity, "text_tokens": text_tokens,
        "structural_tokens": structural_written,
        # total_utf8_bytes counts every scanned document, including the ones whose
        # tail fell outside the last full sequence; consumed_utf8_bytes is what this
        # arm actually reads (== bytes_cum[-1]) and is the BPB denominator.
        "total_utf8_bytes": total_utf8_bytes, "consumed_utf8_bytes": consumed_utf8_bytes,
        "bytes_per_token": consumed_utf8_bytes / capacity, "ids_sha256": ids_sha256,
        "doc_order_sha256": doc_order_sha256, "dropped_tail_tokens": dropped_tail_tokens,
        "elapsed_seconds": elapsed,
        "n_docs": n_docs, "max_docs": args.max_docs, "seed": args.seed,
        "train_path": str(train_path), "train_sha256": train_sha256,
        "endoftext_literal_hits": endoftext_literal_hits,
        "full_corpus_token_check": full_corpus_token_check,
    }
    write_json(meta_path, meta)
    print(f"[{args.arm}] wrote {ids_path} ({ids_sha256[:16]}...), {bytes_path}, {meta_path} "
          f"in {elapsed:.1f}s", flush=True)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--out", default=str(ROOT / "lab/data/v3"))
    parser.add_argument("--max-docs", type=int, default=None,
                        help="limit to the first N documents of --train (for self-test); "
                             "omit to pack the full file. Hard assertion 3 (total_tokens "
                             "+-1%%) is only defined for a full run and is skipped here.")
    parser.add_argument("--expect-ids-sha256", default=None,
                        help="crash unless the packed ids.u16 has this sha256 (decision "
                             "doc S2.2: re-pack on the training machine and assert equality)")
    args = parser.parse_args()
    if args.max_docs is not None and args.max_docs <= 0:
        parser.error("--max-docs must be positive")
    if args.seq_len < 3:
        parser.error("--seq-len must be at least 3 (BOS + >=1 text token + EOS)")
    result = pack(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0)
