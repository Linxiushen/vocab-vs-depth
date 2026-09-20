"""Dataset over lab/pack_corpus.py output (decision doc S1.3, as corrected by
the P2 interface contract handed to the pack/train split of this experiment).

Decision doc S1.3 illustrates the dataset as `x, y = ids[:-1], ids[1:]` with
511 supervised targets. That snippet predates the confirmed fact that
minimind/model/model_minimind.py already shifts internally (logits[...,:-1]
vs labels[...,1:], lines 245-253): feeding it a pre-shifted (x, y) pair would
shift a second time. So this dataset returns the full seq_len ids unshifted;
the training loop is expected to pass the same tensor as both input_ids and
labels and let the model do the one and only shift. This also matches the
"do not mask anything" requirement below, since there is no separate labels
tensor to mask -- see PACKED_DATASET_NOTE at the bottom of this module.

No -100 masking. Upstream's `labels[input_ids == pad_id] = -100` exists to
hide padding, and pad_id (0) is the same id as the literal string
"<|endoftext|>" would encode to in running text (measured 0 occurrences in
lab/data/v3/train.jsonl on both arms, 2026-09-18 -- the hazard is structural,
not observed, and pack_corpus.py keeps counting it as a regression guard).
Packed sequences have zero padding by construction, so that line could only
ever fire on real, meaningful text tokens here -- it must not be reintroduced.

Why the integrity check on open: pack_corpus.py writes a full-size ids.u16 up
front, so a run killed half-way used to leave a plausible-looking array whose
tail was all zeros. pack_corpus.py now stages that array in a .tmp file and
writes meta.json last, and this module refuses to open a prefix whose ids.u16
does not match meta's n_seq/seq_len and ids_sha256. That is also the carrier
for decision doc S2.2's "re-pack on the training machine and assert sha256
equality": the assertion now happens every time the data is opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import sha256_file


class PackedBlocks(Dataset):
    def __init__(self, prefix, verify_sha256=True):
        prefix = str(prefix)
        self.prefix = prefix
        self.ids_path = Path(prefix + ".ids.u16")
        self.bytes_path = Path(prefix + ".bytes_cum.i64")
        self.meta_path = Path(prefix + ".meta.json")
        for path in (self.ids_path, self.bytes_path, self.meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"PackedBlocks({prefix!r}): missing {path}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.n_seq, self.seq_len = self.meta["n_seq"], self.meta["seq_len"]
        expected_bytes = self.n_seq * self.seq_len * 2
        actual_bytes = self.ids_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"PackedBlocks({prefix!r}): {self.ids_path} is {actual_bytes} bytes, "
                f"meta says n_seq={self.n_seq} x seq_len={self.seq_len} x 2 = "
                f"{expected_bytes} -- truncated or mismatched pack")
        expected_cum = self.n_seq * 8
        actual_cum = self.bytes_path.stat().st_size
        if actual_cum != expected_cum:
            raise ValueError(
                f"PackedBlocks({prefix!r}): {self.bytes_path} is {actual_cum} bytes, "
                f"expected n_seq={self.n_seq} x 8 = {expected_cum}")
        self.verified = False
        if verify_sha256:
            actual_sha = sha256_file(self.ids_path)
            if actual_sha != self.meta["ids_sha256"]:
                raise ValueError(
                    f"PackedBlocks({prefix!r}): {self.ids_path} sha256 {actual_sha} != "
                    f"{self.meta['ids_sha256']} recorded in {self.meta_path.name} -- the "
                    f"array was interrupted, corrupted, or does not belong to this meta")
            self.verified = True
        self._ids = None
        self._bytes_cum = None

    # np.memmap inherits ndarray.__reduce__, so keeping one as an attribute would
    # serialize the whole array (605 MB per worker for the full arm-A pack) on any
    # platform where DataLoader spawns instead of forks -- macOS included.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ids"] = None
        state["_bytes_cum"] = None
        return state

    @property
    def ids(self):
        if self._ids is None:
            self._ids = np.memmap(self.ids_path, mode="r", dtype=np.uint16,
                                  shape=(self.n_seq, self.seq_len))
        return self._ids

    @property
    def bytes_cum(self):
        if self._bytes_cum is None:
            self._bytes_cum = np.memmap(self.bytes_path, mode="r", dtype=np.int64,
                                        shape=(self.n_seq,))
        return self._bytes_cum

    def __len__(self):
        return self.n_seq

    def __getitem__(self, i):
        if i < 0 or i >= self.n_seq:
            raise IndexError(i)
        ids = torch.from_numpy(np.asarray(self.ids[i], dtype=np.int64))
        cum = self.bytes_cum
        bytes_delta = int(cum[i]) if i == 0 else int(cum[i] - cum[i - 1])
        return ids, bytes_delta

    def summary(self):
        return dict(self.meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="e.g. lab/data/v3/packed_A")
    parser.add_argument("--skip-sha256", action="store_true",
                        help="skip the ids.u16 sha256 check (size checks still run)")
    args = parser.parse_args()
    ds = PackedBlocks(args.prefix, verify_sha256=not args.skip_sha256)
    print(f"{args.prefix}: {len(ds)} sequences, ids_sha256 "
          f"{'verified' if ds.verified else 'NOT checked (--skip-sha256)'}")
    print(json.dumps(ds.summary(), ensure_ascii=False, indent=2))
    ids0, delta0 = ds[0]
    print(f"item 0: ids.shape={tuple(ids0.shape)} dtype={ids0.dtype} bytes_delta={delta0}")
    ids_last, delta_last = ds[len(ds) - 1]
    print(f"item {len(ds) - 1}: ids.shape={tuple(ids_last.shape)} bytes_delta={delta_last}")


# PACKED_DATASET_NOTE: the training loop (owned by the run_pretrain.py lane)
# should do, per sequence, roughly:
#   ids, bytes_delta = packed_blocks[i]                # int64[seq_len]
#   input_ids = labels = ids.unsqueeze(0)              # same tensor, no shift
#   out = model(input_ids=input_ids, labels=labels)     # model shifts internally
# and accumulate bytes_delta for train_bits_per_byte (decision doc S1.4(b)).
# lab/upstream-pretrain.patch carries its own copy of this class (it must apply
# to a clean minimind checkout); the ids_sha256/size checks above belong there
# too, otherwise the copy that training actually runs stays unprotected.
