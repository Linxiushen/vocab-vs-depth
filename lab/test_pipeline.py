"""Regression tests for data leakage, reserved tokens, and complete BPB scoring."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoTokenizer

from common import ROOT, read_texts, text_key, write_texts
from eval_bpb import score_ids
from prepare_experiment import prepare
from train_tokenizer_16k import derive_prefix, train


class PipelineTests(unittest.TestCase):
    def test_content_duplicates_and_smoke_are_excluded_from_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_texts(root / "source", ["seen", "new", "hold", "hold", "train"])
            write_texts(root / "holdout", ["seen", "hold"])
            write_texts(root / "smoke", ["seen"])
            report = prepare(root / "source", root / "holdout", root / "smoke", root / "out", 2)
            self.assertEqual(report["heldout_source_rows"], 2)
            self.assertEqual(list(read_texts(root / "out/val.jsonl")), ["hold"])
            self.assertNotIn("hold", list(read_texts(root / "out/train.jsonl")))
            prepare(root / "source", root / "holdout", root / "smoke", root / "again", 2)
            self.assertEqual((root / "out/tokenizer_train.jsonl").read_bytes(),
                             (root / "again/tokenizer_train.jsonl").read_bytes())

    def test_score_all_tokens_including_one_token_tail(self):
        seen_targets = []

        class UniformModel:
            def __call__(self, inputs):
                seen_targets.extend(inputs[0, 1:].tolist())
                return SimpleNamespace(logits=torch.zeros(1, inputs.shape[1], 16))

        ids = [3, 4, 5, 6, 7, 8, 9]
        nll, chunks = score_ids(UniformModel(), ids, 1, "cpu", 5)
        self.assertEqual(seen_targets, ids)
        self.assertEqual(chunks, 3)
        self.assertAlmostEqual(nll, len(ids) * math.log(16), places=5)

    def test_tokenizer_reload_preserves_delimiters_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = ["你好，世界。 tokenizer test 123\n", "学习语言模型的词表。"] * 30
            write_texts(root / "data", corpus)
            train(root / "data", root / "tokenizer", 320)
            tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
            upstream = json.loads((ROOT / "minimind/model/tokenizer.json").read_text())
            saved = json.loads((root / "tokenizer/tokenizer.json").read_text())
            self.assertEqual(upstream["added_tokens"], saved["added_tokens"])
            for text in corpus + ["罕见字𠮷，空格  \n\t", "<think>hi</think><tool_call>x</tool_call>"]:
                self.assertEqual(tok.decode(tok.encode(text), skip_special_tokens=True), text)
            derive_prefix(root / "tokenizer", root / "prefix", 300)
            train(root / "data", root / "independent", 300)
            prefix = AutoTokenizer.from_pretrained(root / "prefix", local_files_only=True)
            independent = AutoTokenizer.from_pretrained(root / "independent", local_files_only=True)
            self.assertEqual(prefix.get_vocab(), independent.get_vocab())
            for text in corpus:
                self.assertEqual(prefix.encode(text), independent.encode(text))


if __name__ == "__main__":
    unittest.main()
