#!/usr/bin/env bash
# Apply this project's small modification to the pinned upstream MiniMind checkout.
#
# Upstream is a git submodule pinned at a3c7b01cc004d5de86aea961f20bf1e638e7c09e.
# The only change is lab/upstream-pretrain.patch, touching trainer/train_pretrain.py
# and trainer/trainer_utils.py: --vocab_size/--tokenizer_path plus a consistency
# check, and (决策文档 §1.4/§1.5) a --packed fixed-token-budget training path with
# its own padding-free Dataset, a WSD learning-rate schedule alongside upstream's
# unchanged cosine one, per-optimizer-update run_ledger.jsonl accounting, and
# --save_steps_frac checkpointing.
# Idempotent: running twice is a no-op.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/lab/upstream-pretrain.patch"
cd "$ROOT/minimind"

if [ ! -f trainer/train_pretrain.py ]; then
  echo "minimind submodule is empty. Run: git submodule update --init" >&2
  exit 1
fi
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
  exit 0
fi
git apply --check "$PATCH"
git apply "$PATCH"
echo "applied $(basename "$PATCH") to minimind@$(git rev-parse --short HEAD)"
