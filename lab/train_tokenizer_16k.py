"""Train byte-level BPE on a pre-sampled training-only JSONL corpus.

Use the same corpus and pre-tokenizer for both vocabulary sizes. Preserve the
upstream token IDs AND special flags; tool/think delimiters must survive decode.
GGUF conversion is a separate test, not guaranteed by the pre-tokenizer regex.
"""
import argparse
import json
from pathlib import Path
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from common import ROOT, read_texts, sha256_file, write_json


def train(data, out, vocab, reference=ROOT / "minimind/model"):
    out, reference = Path(out), Path(reference)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    original = json.loads((reference / "tokenizer.json").read_text())
    added = sorted(original["added_tokens"], key=lambda item: item["id"])
    if [item["id"] for item in added] != list(range(len(added))):
        raise ValueError("Reference reserved IDs must be contiguous from zero")
    if vocab < 256 + len(added):
        raise ValueError("Vocabulary cannot hold byte alphabet and reserved tokens")
    source_tokenizer = Tokenizer.from_file(str(reference / "tokenizer.json"))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = source_tokenizer.pre_tokenizer
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[item["content"] for item in added],
    )
    stats = {"documents": 0, "utf8_bytes": 0}

    def texts():
        for text in read_texts(data):
            stats["documents"] += 1
            stats["utf8_bytes"] += len(text.encode("utf-8"))
            if stats["documents"] % 50_000 == 0:
                print(f"BPE ingested {stats['documents']:,} documents", flush=True)
            yield text

    start = time.perf_counter()
    print(f"Training vocab={vocab} from {data}", flush=True)
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if not stats["documents"] or tokenizer.get_vocab_size() != vocab:
        raise ValueError("Empty input or corpus too small for requested vocabulary")

    serialized = json.loads(tokenizer.to_str())
    serialized["added_tokens"] = added
    for item in added:
        if serialized["model"]["vocab"].get(item["content"]) != item["id"]:
            raise AssertionError("Reserved token ID changed")
    out.mkdir(parents=True)
    write_json(out / "tokenizer.json", serialized)
    config = json.loads((reference / "tokenizer_config.json").read_text())
    config["added_tokens_decoder"] = {
        str(item["id"]): {key: value for key, value in item.items() if key != "id"}
        for item in added
    }
    write_json(out / "tokenizer_config.json", config)
    loaded = AutoTokenizer.from_pretrained(out, local_files_only=True)
    if len(loaded) != vocab or (loaded.pad_token_id, loaded.bos_token_id,
                                loaded.eos_token_id) != (0, 1, 2):
        raise AssertionError("Reload changed vocabulary size or special IDs")
    for item in added:
        if loaded.encode(item["content"], add_special_tokens=False) != [item["id"]]:
            raise AssertionError(f"Reserved token split: {item['content']}")
    sample = '<think>test</think><tool_call>{"name":"test"}</tool_call>'
    if loaded.decode(loaded.encode(sample), skip_special_tokens=True) != sample:
        raise AssertionError("Decode dropped tool or think delimiters")
    report = {
        "vocab_size": vocab, "data": str(Path(data).resolve()),
        "data_sha256": sha256_file(data), **stats,
        "reference_tokenizer_sha256": sha256_file(reference / "tokenizer.json"),
        "tokenizer_sha256": sha256_file(out / "tokenizer.json"),
        "pre_tokenizer": serialized["pre_tokenizer"],
        "elapsed_seconds": time.perf_counter() - start,
        "gguf_export": "not tested",
    }
    write_json(out / "training_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def derive_prefix(source, out, vocab):
    """Use an earlier stopping point in the same learned BPE merge hierarchy."""
    start = time.perf_counter()
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    serialized = json.loads((source / "tokenizer.json").read_text())
    original_vocab = serialized["model"]["vocab"]
    if not len(serialized["added_tokens"]) + 256 <= vocab < len(original_vocab):
        raise ValueError("Prefix must retain reserved tokens and every byte token")
    retained = {token: index for token, index in original_vocab.items() if index < vocab}
    serialized["model"]["vocab"] = retained
    serialized["model"]["merges"] = [pair for pair in serialized["model"]["merges"]
                                       if pair[0] + pair[1] in retained]
    out.mkdir(parents=True)
    write_json(out / "tokenizer.json", serialized)
    write_json(out / "tokenizer_config.json", json.loads((source / "tokenizer_config.json").read_text()))
    loaded = AutoTokenizer.from_pretrained(out, local_files_only=True)
    if len(loaded) != vocab:
        raise AssertionError("Derived tokenizer size changed on reload")
    parent = json.loads((source / "training_manifest.json").read_text())
    report = {**parent, "vocab_size": vocab,
              "derivation": "prefix of the same learned BPE merge hierarchy",
              "parent_tokenizer_sha256": sha256_file(source / "tokenizer.json"),
              "tokenizer_sha256": sha256_file(out / "tokenizer.json"),
              "parent_training_elapsed_seconds": parent["elapsed_seconds"],
              "elapsed_seconds": time.perf_counter() - start}
    write_json(out / "training_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--derive-from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--reference", type=Path, default=ROOT / "minimind/model")
    args = parser.parse_args()
    if args.derive_from:
        derive_prefix(args.derive_from, args.out, args.vocab)
    else:
        train(args.data, args.out, args.vocab, args.reference)
