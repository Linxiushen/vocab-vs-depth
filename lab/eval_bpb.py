"""Score every byte of fixed held-out documents with a recorded chunk protocol.

Two protocols, selected by --protocol:

  v2 (default, frozen): each chunk gets BOS and at most max_seq_len-2 body
  tokens, chosen by greedily windowing THAT run's own --tokenizer over each
  document independently. All body tokens are scored exactly once; no
  tokenizer-dependent truncation or decoded-prefix byte counts. Context
  resets at chunk boundaries. This is the historical protocol and its output
  on lab/data/v2/val.jsonl is a regression fixture (see the module tests /
  self-test below) -- evaluate() and score_ids() below must not change by a
  single byte.

  v3 (决策-20260918.md §1.2 D4): chunk boundaries are fixed once, in
  characters, by lab/build_eval_chunks_v3.py using the smaller/nested
  tokenizer's own <=510-token budget, and every arm scores the identical
  [cs, ce) text span (see --chunks). This removes a confound v2 has: two
  tokenizers that need different numbers of tokens for the same text reset
  context at different rates on identical documents, which --mirror-confusion
  below quantifies. v3 boundaries are only valid for the tokenizer family
  they were built with -- see evaluate_v3()'s docstring.

--mirror-confusion is a separate, model-free diagnostic: it counts how many
independent v2-style chunks --tokenizer-a and --tokenizer-b each need on
--val, which is the evidence cited in 决策-20260918.md §2.3 for why v3 exists.

--se-regression is the third model-free mode: given two eval JSONs it reports
the delta-method standard errors of each arm's BPB, of their unpaired
difference, and of the paired (per-document) difference, i.e. the precision
gate 决策-20260918.md §三 requires be reproduced before any 64M run starts:

    python lab/eval_bpb.py --se-regression \
        --a lab/results/random-eval-v2.json --b lab/results/smoke-eval-v2.json \
        --expect-unpaired-se 0.028305 --expect-paired-se 0.011925 --expect-gain 2.373
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from common import ROOT, read_texts, sha256_file, write_json

sys.path.insert(0, str(ROOT / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def score_ids(model, ids, bos, device, max_seq_len):
    if max_seq_len < 3:
        raise ValueError("max_seq_len must be at least 3")
    total = 0.0
    chunks = 0
    with torch.inference_mode():
        for start in range(0, len(ids), max_seq_len - 2):
            body = ids[start:start + max_seq_len - 2]
            inputs = torch.tensor([[bos] + body], device=device)
            logits = model(inputs).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, inputs[0, 1:], reduction="sum")
            total += loss.item()
            chunks += 1
    return total, chunks


def evaluate(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    vocab = len(tok) if args.vocab_size is None else args.vocab_size
    if vocab != len(tok):
        raise ValueError("Model vocabulary size differs from tokenizer")
    torch.manual_seed(args.seed)
    config = MiniMindConfig(hidden_size=args.hidden_size,
                           num_hidden_layers=args.num_hidden_layers,
                           vocab_size=vocab, bos_token_id=tok.bos_token_id,
                           eos_token_id=tok.eos_token_id)
    model = MiniMindForCausalLM(config)
    if args.weight:
        state = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    rows = []
    total_nll = total_bytes = total_tokens = 0
    for index, text in enumerate(read_texts(args.val)):
        if args.limit and index >= args.limit:
            break
        ids = tok.encode(text, add_special_tokens=False)
        if not ids or tok.decode(ids, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False) != text:
            raise ValueError(f"Document {index} failed lossless tokenization")
        nll, chunks = score_ids(model, ids, tok.bos_token_id, args.device, args.max_seq_len)
        nbytes = len(text.encode("utf-8"))
        total_nll += nll
        total_bytes += nbytes
        total_tokens += len(ids)
        rows.append({"index": index, "utf8_bytes": nbytes, "tokens": len(ids),
                     "nll_nats": nll, "chunks": chunks})
        if len(rows) % 250 == 0:
            print(f"Evaluated {len(rows)} documents", flush=True)
    if not total_tokens or not total_bytes:
        raise ValueError("No evaluable documents")
    loss = total_nll / total_tokens
    result = {
        "protocol": "all_body_tokens_bos_per_chunk_v2",
        "max_seq_len": args.max_seq_len,
        "documents": len(rows), "utf8_bytes": total_bytes,
        "tokens": total_tokens, "nll_nats": total_nll,
        "loss_nats_per_token": loss, "ppl": math.exp(loss),
        "bpb": total_nll / (math.log(2) * total_bytes),
        "val_sha256": sha256_file(args.val),
        "weight_sha256": sha256_file(args.weight) if args.weight else None,
        "tokenizer_sha256": sha256_file(Path(args.tokenizer) / "tokenizer.json"),
        "initialization": "checkpoint" if args.weight else "random",
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": args.device, "dtype": str(next(model.parameters()).dtype),
        "per_document": rows,
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_document"}, indent=2), flush=True)
    return result


def _build_model(tok, args):
    """Model construction shared by evaluate_v3(); evaluate() keeps its own
    copy inline so the frozen v2 path never depends on this function."""
    vocab = len(tok) if args.vocab_size is None else args.vocab_size
    if vocab != len(tok):
        raise ValueError("Model vocabulary size differs from tokenizer")
    torch.manual_seed(args.seed)
    config = MiniMindConfig(hidden_size=args.hidden_size,
                           num_hidden_layers=args.num_hidden_layers,
                           vocab_size=vocab, bos_token_id=tok.bos_token_id,
                           eos_token_id=tok.eos_token_id)
    model = MiniMindForCausalLM(config)
    if args.weight:
        state = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    return model


def score_v3_chunk(model, body_ids, bos, device, max_seq_len):
    # max_seq_len - 2 rather than - 1: v2's score_ids() windows bodies at
    # max_seq_len - 2 too, so both protocols feed the model sequences of the
    # same maximum length and a v2/v3 pair is never compared across different
    # effective context sizes.
    if len(body_ids) > max_seq_len - 2:
        raise ValueError(
            f"v3 chunk needs {len(body_ids)} body tokens but max_seq_len={max_seq_len} "
            f"allows only {max_seq_len - 2} (BOS + body, the same window v2 uses). v3 "
            f"chunk boundaries are only "
            f"valid for the tokenizer family they were built with -- a verified nested "
            f"merge-table pair where the shorter tokenizer's budget provably bounds the "
            f"longer one's token count -- not for an unrelated tokenizer that merely "
            f"shares the same nominal vocab_size (e.g. upstream MiniMind's tokenizer vs "
            f"bpe_6400: same vocab_size=6400, different merge table, no such guarantee)."
        )
    with torch.inference_mode():
        inputs = torch.tensor([[bos] + body_ids], device=device)
        logits = model(inputs).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, inputs[0, 1:], reduction="sum")
    return loss.item()


def load_chunks_meta(chunks_path, val_path, max_seq_len, tokenizer_path):
    """Raise unless the sidecar manifest proves these chunks are the intact,
    current artifact built from exactly this --val at a compatible budget.

    Without this, a truncated, hand-edited or simply stale chunk file makes
    evaluate_v3() silently score whatever subset of documents it happens to
    mention and still emit a normal-looking JSON: two arms fed the same bad
    file agree on index and utf8_bytes, so lab/power_analysis.py's guards stay
    quiet too, and a headline BPB difference ends up computed over the wrong
    document set with only result["documents"] as a witness. The validation
    set is still being frozen (决策-20260918.md §1.2 D1), so rebuilt val files
    are expected, which is exactly when this trap gets sprung.
    """
    path = Path(chunks_path)
    meta_path = path.with_name(path.stem + ".meta.json")
    if not meta_path.is_file():
        raise ValueError(
            f"Missing sidecar manifest {meta_path}; --chunks must be an artifact of "
            f"lab/build_eval_chunks_v3.py, which writes the two files together"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    digest = sha256_file(path)
    if meta.get("out_sha256") != digest:
        raise ValueError(
            f"{path} has sha256 {digest} but {meta_path} records "
            f"{meta.get('out_sha256')!r}: the chunk file is truncated or edited, or the "
            f"manifest is stale. Rebuild with lab/build_eval_chunks_v3.py; do not score a "
            f"chunk set nobody can reconstruct."
        )
    val_digest = sha256_file(val_path)
    if meta.get("val_sha256") != val_digest:
        raise ValueError(
            f"--val {val_path} has sha256 {val_digest} but these chunks were built from "
            f"{meta.get('val')!r} with sha256 {meta.get('val_sha256')!r}. Chunk boundaries are "
            f"character offsets into specific documents; applying them to a different (even "
            f"slightly re-derived) validation file scores the wrong text."
        )

    budget = meta.get("budget")
    if not isinstance(budget, int) or budget < 1:
        raise ValueError(f"{meta_path} records an unusable budget {budget!r}")
    if budget > max_seq_len - 2:
        raise ValueError(
            f"These chunks were built with budget={budget}, but --max_seq_len={max_seq_len} "
            f"leaves only {max_seq_len - 2} body-token slots (BOS + body, v2's window "
            f"convention). Either score them at the max_seq_len they were built for, or "
            f"rebuild the chunks -- silently evaluating at a narrower context would make the "
            f"number incomparable to every other run."
        )
    for field in ("documents", "chunks"):
        if not isinstance(meta.get(field), int) or meta[field] < 1:
            raise ValueError(f"{meta_path} records an unusable {field}={meta.get(field)!r}")

    # v3 fixes boundaries at A's token edges, so B restarts mid-token at every
    # cut and pays a small, systematically A-favouring token overhead that v3
    # does NOT remove. The builder measures it; requiring it here means no v3
    # number can be produced without that residual travelling alongside it in
    # the result JSON below.
    if not isinstance(meta.get("residual_boundary_token_overhead"), dict):
        raise ValueError(
            f"{meta_path} has no residual_boundary_token_overhead: rebuild with the current "
            f"lab/build_eval_chunks_v3.py. v3's leftover, A-favouring boundary asymmetry has "
            f"to be reportable next to the BPB it affects, not left as folklore."
        )

    # The nesting proof that makes these boundaries safe was checked at build
    # time for exactly two tokenizers. score_v3_chunk() below catches a foreign
    # tokenizer only when some chunk happens to overflow the budget; one that
    # fits everywhere would otherwise produce a perfectly normal-looking BPB on
    # a protocol whose precondition was never verified for it.
    family = {meta.get("tokenizer_a_sha256"): meta.get("tokenizer_a"),
              meta.get("tokenizer_b_sha256"): meta.get("tokenizer_b")}
    tokenizer_digest = sha256_file(Path(tokenizer_path) / "tokenizer.json")
    if tokenizer_digest not in family:
        raise ValueError(
            f"--tokenizer {tokenizer_path} (tokenizer.json sha256 {tokenizer_digest}) is not "
            f"one of the two tokenizers these chunks were built and verified against: "
            f"{meta.get('tokenizer_a')!r} ({meta.get('tokenizer_a_sha256')!r}) and "
            f"{meta.get('tokenizer_b')!r} ({meta.get('tokenizer_b_sha256')!r}). v3 boundaries "
            f"are only valid inside that verified nested merge-table family -- a tokenizer "
            f"merely sharing the nominal vocab_size (upstream MiniMind's vs bpe_6400) has no "
            f"len_B<=len_A guarantee, and if its chunks happen to fit it would yield a "
            f"plausible-looking number with no protocol behind it."
        )
    return meta, digest


def evaluate_v3(args):
    """Score lab/build_eval_chunks_v3.py's fixed common-text chunks.

    Each chunk row's [cs, ce) span is re-encoded fresh with --tokenizer and
    scored as [BOS] + body, every body token, no EOS -- the denominator is
    the row's own recorded utf8_bytes, cross-checked against a fresh
    text.encode('utf-8') rather than trusted blindly. Chunks are summed back
    into one row per document (nll_nats, utf8_bytes, tokens, chunk count) so
    lab/power_analysis.py's paired bootstrap can still match documents across
    two v3 evaluations the same way it does for v2.

    Chunk boundaries only bound the tokenizer they were built from (see
    lab/build_eval_chunks_v3.py): the guarantee that a <=510-A-token chunk is
    also <=510 B-tokens rests on A's merge table being a verified prefix of
    B's. Scoring with a tokenizer outside that nested family -- same nominal
    vocab_size, unrelated merges, e.g. upstream MiniMind's tokenizer against
    chunks built from bpe_6400 -- has no such guarantee and can legitimately
    need more than the max_seq_len-2 body tokens a chunk is allowed.
    score_v3_chunk() raises in that case rather than silently re-splitting the
    chunk, which would quietly turn the eval back into a v2-shaped protocol.

    The chunk file is not taken on trust either: load_chunks_meta() refuses to
    start unless its sidecar manifest matches the file's sha256 and the
    --val's, and every --val document must appear in it -- see that function
    for why a partial chunk file is the dangerous case.
    """
    meta, chunks_sha256 = load_chunks_meta(args.chunks, args.val, args.max_seq_len,
                                           args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = _build_model(tok, args)
    texts = list(read_texts(args.val))

    by_doc = {}
    n_rows = 0
    with open(args.chunks, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_doc.setdefault(row["doc"], []).append(row)
            n_rows += 1

    if n_rows != meta["chunks"] or len(by_doc) != meta["documents"]:
        raise ValueError(
            f"--chunks holds {n_rows} rows over {len(by_doc)} documents but its manifest "
            f"records {meta['chunks']} rows over {meta['documents']} documents"
        )
    if len(texts) != meta["documents"]:
        raise ValueError(
            f"--val has {len(texts)} documents but these chunks were built over "
            f"{meta['documents']}"
        )
    if sorted(by_doc) != list(range(len(texts))):
        missing = sorted(set(range(len(texts))) - set(by_doc))
        raise ValueError(
            f"--chunks does not cover every --val document: {len(missing)} missing, "
            f"e.g. {missing[:5]}. v3 scores every byte of the held-out set; a partial "
            f"chunk file would quietly shrink the evaluation instead of failing."
        )

    doc_indices = sorted(by_doc)
    if args.limit:
        doc_indices = doc_indices[:args.limit]

    doc_rows = []
    total_nll = total_bytes = total_tokens = 0
    for doc in doc_indices:
        if doc < 0 or doc >= len(texts):
            raise ValueError(f"--chunks references doc {doc} but --val has only {len(texts)} documents")
        text = texts[doc]
        chunk_rows = sorted(by_doc[doc], key=lambda row: row["chunk"])
        cursor = 0
        doc_nll = doc_bytes = doc_tokens = 0
        for row in chunk_rows:
            cs, ce = row["cs"], row["ce"]
            if cs != cursor:
                raise ValueError(f"doc {doc} chunk {row['chunk']}: starts at {cs}, expected {cursor}")
            span = text[cs:ce]
            nbytes = len(span.encode("utf-8"))
            if nbytes != row["utf8_bytes"]:
                raise ValueError(f"doc {doc} chunk {row['chunk']}: recorded utf8_bytes "
                                 f"{row['utf8_bytes']} != recomputed {nbytes}")
            body_ids = tok.encode(span, add_special_tokens=False)
            # Same guard as v2's evaluate(): the numerator is counted over
            # tokens and the denominator over the span's raw bytes, so a
            # tokenizer that cannot reproduce the span would be billing NLL
            # for bytes it never modelled.
            if not body_ids or tok.decode(body_ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False) != span:
                raise ValueError(f"doc {doc} chunk {row['chunk']} failed lossless tokenization")
            doc_nll += score_v3_chunk(model, body_ids, tok.bos_token_id, args.device, args.max_seq_len)
            doc_bytes += nbytes
            doc_tokens += len(body_ids)
            cursor = ce
        if cursor != len(text):
            raise ValueError(f"doc {doc}: chunks cover [0,{cursor}) but document is {len(text)} chars")
        total_nll += doc_nll
        total_bytes += doc_bytes
        total_tokens += doc_tokens
        doc_rows.append({"index": doc, "utf8_bytes": doc_bytes, "tokens": doc_tokens,
                         "nll_nats": doc_nll, "chunks": len(chunk_rows)})
        if len(doc_rows) % 250 == 0:
            print(f"Evaluated {len(doc_rows)} documents", flush=True)

    if not total_tokens or not total_bytes:
        raise ValueError("No evaluable documents")
    loss = total_nll / total_tokens
    result = {
        "protocol": "common_text_chunks_bos_per_chunk_v3",
        "max_seq_len": args.max_seq_len,
        "chunks_path": args.chunks,
        "chunks_sha256": chunks_sha256,
        "chunk_budget": meta["budget"],
        "residual_boundary_token_overhead": meta["residual_boundary_token_overhead"],
        "val_documents": len(texts),
        "limit": args.limit,
        "documents": len(doc_rows), "utf8_bytes": total_bytes,
        "tokens": total_tokens, "nll_nats": total_nll,
        "loss_nats_per_token": loss, "ppl": math.exp(loss),
        "bpb": total_nll / (math.log(2) * total_bytes),
        "val_sha256": sha256_file(args.val),
        "weight_sha256": sha256_file(args.weight) if args.weight else None,
        "tokenizer_sha256": sha256_file(Path(args.tokenizer) / "tokenizer.json"),
        "initialization": "checkpoint" if args.weight else "random",
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": args.device, "dtype": str(next(model.parameters()).dtype),
        "per_document": doc_rows,
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_document"}, indent=2), flush=True)
    return result


def mirror_confusion(args):
    """Count v2-protocol chunks --tokenizer-a and --tokenizer-b each need on
    --val, independently, with no model involved.

    Reproduces 决策-20260918.md §2.3's "v2 镜像混淆" evidence: v2 (evaluate())
    splits each document into ceil(tokens/(max_seq_len-2)) windows using that
    run's OWN tokenizer, so on the identical documents the tokenizer that
    needs more tokens per byte resets its BOS-anchored context more often --
    a tokenizer-length artifact, not a fact about compression. This is why
    v3's common text chunking (lab/build_eval_chunks_v3.py) exists.
    """
    tok_a = AutoTokenizer.from_pretrained(args.tokenizer_a, local_files_only=True)
    tok_b = AutoTokenizer.from_pretrained(args.tokenizer_b, local_files_only=True)
    window = args.max_seq_len - 2
    if window < 1:
        raise ValueError("max_seq_len must be at least 3")
    documents = tokens_a = tokens_b = chunks_a = chunks_b = 0
    total_bytes = 0
    multi_a = multi_b = 0
    multi_bytes_a = multi_bytes_b = 0
    for index, text in enumerate(read_texts(args.val)):
        if args.limit and index >= args.limit:
            break
        documents += 1
        n_a = len(tok_a.encode(text, add_special_tokens=False))
        n_b = len(tok_b.encode(text, add_special_tokens=False))
        if not n_a or not n_b:
            raise ValueError("Tokenizer produced zero tokens for a nonempty document")
        nbytes = len(text.encode("utf-8"))
        tokens_a += n_a
        tokens_b += n_b
        total_bytes += nbytes
        windows_a, windows_b = -(-n_a // window), -(-n_b // window)
        chunks_a += windows_a
        chunks_b += windows_b
        # 决策-20260918.md §2.3 also cites WHERE the extra resets live: the
        # documents long enough to be split at all. Those two figures were
        # only ever quoted in prose, so record them in the artifact too.
        if windows_a > 1:
            multi_a += 1
            multi_bytes_a += nbytes
        if windows_b > 1:
            multi_b += 1
            multi_bytes_b += nbytes
    if not documents:
        raise ValueError("No evaluable documents")
    return {
        "protocol": "v2_mirror_confusion_evidence",
        "val": args.val, "val_sha256": sha256_file(args.val),
        "max_seq_len": args.max_seq_len, "window_tokens": window,
        "documents": documents, "limit": args.limit,
        "utf8_bytes": total_bytes,
        "tokenizer_a": args.tokenizer_a,
        "tokenizer_a_sha256": sha256_file(Path(args.tokenizer_a) / "tokenizer.json"),
        "tokenizer_b": args.tokenizer_b,
        "tokenizer_b_sha256": sha256_file(Path(args.tokenizer_b) / "tokenizer.json"),
        "tokens_a": tokens_a, "tokens_b": tokens_b,
        "chunks_a": chunks_a, "chunks_b": chunks_b,
        "extra_chunks_a_minus_b": chunks_a - chunks_b,
        "extra_context_resets_pct_of_b": ((chunks_a - chunks_b) / chunks_b * 100
                                          if chunks_b else None),
        "multi_chunk_documents_a": multi_a,
        "multi_chunk_documents_b": multi_b,
        "multi_chunk_bytes_pct_a": multi_bytes_a / total_bytes * 100,
        "multi_chunk_bytes_pct_b": multi_bytes_b / total_bytes * 100,
    }


def ratio_stderr(numerators, denominators, ratio):
    """Delta-method standard error of sum(numerators)/sum(denominators).

    BPB is a ratio of two sums, not a mean of per-document values, so its
    sampling error comes from linearizing the ratio: the per-document
    residual r_i = y_i - R*x_i carries all the variance, and
    Var(R) ~= n*Var(r)/(sum x)^2. lab/power_analysis.py gets the same quantity
    by resampling; this closed form is what 决策-20260918.md §三 quotes, and
    having both lets each check the other (they agree to ~1% at 2000
    resamples).
    """
    n = len(denominators)
    if n < 2:
        raise ValueError("Need at least two documents for a standard error")
    total = sum(denominators)
    if not total:
        raise ValueError("Zero total bytes")
    residuals = [y - ratio * x for y, x in zip(numerators, denominators)]
    mean = sum(residuals) / n
    variance = sum((r - mean) ** 2 for r in residuals) / (n - 1)
    return math.sqrt(n * variance) / total


def se_regression(args):
    """Reproduce 决策-20260918.md §三's precision gate from two eval JSONs.

    The gate exists because the whole 20-run budget is sized on the claim that
    scoring both arms on the SAME held-out documents and differencing
    per-document buys a ~2.37x smaller standard error than comparing two
    independently-estimated BPBs. That claim was a hand calculation in the
    decision doc; this mode makes it a command anyone can re-run, and with
    --expect-* a hard gate that raises rather than prints a wrong number.

    No model is loaded and --val/--chunks/--limit play no part.
    """
    meta_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    meta_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    rows_a, rows_b = meta_a["per_document"], meta_b["per_document"]
    if meta_a.get("protocol") != meta_b.get("protocol"):
        raise ValueError("Evaluations use different protocols; not comparable")
    if [row["index"] for row in rows_a] != [row["index"] for row in rows_b]:
        raise ValueError("Evaluations cover different documents or ordering")
    if [row["utf8_bytes"] for row in rows_a] != [row["utf8_bytes"] for row in rows_b]:
        raise ValueError("Byte counts differ; evaluations scored different text")

    nbytes = [row["utf8_bytes"] for row in rows_a]
    bits_a = [row["nll_nats"] / math.log(2) for row in rows_a]
    bits_b = [row["nll_nats"] / math.log(2) for row in rows_b]
    total_bytes = sum(nbytes)
    if not total_bytes:
        raise ValueError("No evaluable documents")
    bpb_a, bpb_b = sum(bits_a) / total_bytes, sum(bits_b) / total_bytes
    delta = bpb_a - bpb_b

    se_a = ratio_stderr(bits_a, nbytes, bpb_a)
    se_b = ratio_stderr(bits_b, nbytes, bpb_b)
    unpaired = math.hypot(se_a, se_b)
    paired = ratio_stderr([x - y for x, y in zip(bits_a, bits_b)], nbytes, delta)
    if not paired:
        raise ValueError("Paired standard error is zero; the two evaluations are identical")

    result = {
        "protocol": meta_a.get("protocol"),
        "a": {"path": args.a, "bpb": bpb_a, "weight_sha256": meta_a.get("weight_sha256"),
              "initialization": meta_a.get("initialization")},
        "b": {"path": args.b, "bpb": bpb_b, "weight_sha256": meta_b.get("weight_sha256"),
              "initialization": meta_b.get("initialization")},
        "documents": len(nbytes), "utf8_bytes": total_bytes,
        "bpb_difference_a_minus_b": delta,
        "se_a": se_a, "se_b": se_b,
        "unpaired_se": unpaired,
        "paired_delta_se": paired,
        "precision_gain_paired_over_unpaired": unpaired / paired,
        "method": "delta-method (linearized ratio) standard errors over documents",
        "covers": "measurement noise on this fixed validation set only",
        "does_not_cover": "seed-to-seed training variance; needs repeated training runs",
    }

    expectations = (("unpaired_se", unpaired, args.expect_unpaired_se),
                    ("paired_delta_se", paired, args.expect_paired_se),
                    ("precision_gain_paired_over_unpaired", unpaired / paired, args.expect_gain))
    checks, failures = [], []
    for name, value, expected in expectations:
        if expected is None:
            continue
        deviation = abs(value - expected) / abs(expected)
        passed = deviation <= args.tolerance
        checks.append({"name": name, "value": value, "expected": expected,
                       "relative_deviation": deviation, "passed": passed})
        if not passed:
            failures.append(f"{name}: got {value!r}, expected {expected!r} "
                            f"(relative deviation {deviation:.3e} > {args.tolerance:.3e})")
    result["gate"] = checks or None
    if failures:
        raise ValueError("Precision-gate regression failed -- " + "; ".join(failures))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weight")
    weights.add_argument("--random-init", action="store_true")
    parser.add_argument("--tokenizer", default=str(ROOT / "minimind/model"))
    parser.add_argument("--val", help="required for every mode except --se-regression")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every document")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--protocol", choices=["v2", "v3"], default="v2",
                        help="v2 (default) is the frozen regression protocol and must not "
                             "change; v3 scores lab/build_eval_chunks_v3.py's common chunks "
                             "(requires --chunks)")
    parser.add_argument("--chunks", help="eval_chunks_v3.jsonl from build_eval_chunks_v3.py; "
                                         "required for --protocol v3")
    parser.add_argument("--mirror-confusion", action="store_true",
                        help="model-free diagnostic: count how many independent v2-style "
                             "chunks --tokenizer-a and --tokenizer-b each need on --val "
                             "(决策-20260918.md §2.3 evidence for why v3 exists); ignores "
                             "--weight/--random-init/--tokenizer/--protocol/--chunks")
    parser.add_argument("--tokenizer-a", default=str(ROOT / "lab/tokenizers/bpe_6400"),
                        help="only used by --mirror-confusion")
    parser.add_argument("--tokenizer-b", default=str(ROOT / "lab/tokenizers/bpe_16384"),
                        help="only used by --mirror-confusion")
    parser.add_argument("--se-regression", action="store_true",
                        help="model-free: delta-method standard errors of --a and --b's BPB, "
                             "of their unpaired difference and of the paired per-document "
                             "difference (决策-20260918.md §三 precision gate); ignores "
                             "--val/--weight/--random-init/--tokenizer/--protocol/--chunks/--limit")
    parser.add_argument("--a", help="baseline eval JSON; only used by --se-regression")
    parser.add_argument("--b", help="comparison eval JSON; only used by --se-regression")
    parser.add_argument("--expect-unpaired-se", type=float,
                        help="raise unless --se-regression's unpaired SE matches, within --tolerance")
    parser.add_argument("--expect-paired-se", type=float,
                        help="raise unless --se-regression's paired delta SE matches")
    parser.add_argument("--expect-gain", type=float,
                        help="raise unless --se-regression's unpaired/paired ratio matches")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="relative tolerance for the --expect-* gates (default 1e-3, which "
                             "accepts the decision doc's rounded literals)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.mirror_confusion and args.se_regression:
        parser.error("--mirror-confusion and --se-regression are separate modes")
    if args.se_regression:
        if not args.a or not args.b:
            parser.error("--se-regression requires --a and --b")
        result = se_regression(args)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        sys.exit(0)
    if not args.val:
        parser.error("the following arguments are required: --val")
    if args.mirror_confusion:
        result = mirror_confusion(args)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        if not args.weight and not args.random_init:
            parser.error("one of the arguments --weight --random-init is required")
        if args.protocol == "v3":
            if not args.chunks:
                parser.error("--protocol v3 requires --chunks")
            evaluate_v3(args)
        else:
            evaluate(args)
