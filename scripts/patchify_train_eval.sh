#!/usr/bin/env bash
#
# patchify_train_eval.sh — the patchify-trunk ablation for HPPNet+.
#
# Same one-shot setup + train + evaluate as the baseline, but swaps the harmonic dilated-conv
# acoustic model (CNNTrunk) for the AuM/ViT-style patch-embedding trunk (PatchTrunk) via the
# `patchify` sacred named config. It sweeps the same three sequence models (lstm / mamba /
# bimamba) — the patch trunk reuses the same seq_model axis — so you get patch_lstm / patch_mamba
# / patch_bimamba as three wandb runs grouped by $EXPERIMENT, comparable to the CNNTrunk runs
# from runpod_train_eval.sh.
#
# All of the environment setup + safeguards + the train/eval loop are shared, unchanged, from
# scripts/lib/common.sh — this script only sets the patchify-specific knobs. That is the pattern:
# one experiment == one thin script; never bake a new axis into another experiment's script.
#
# Usage:
#   export WANDB_API_KEY=xxxxxxxx           # from https://wandb.ai/authorize
#   bash scripts/patchify_train_eval.sh
#
# Overridable via env just like the baseline, e.g. a cheap end-to-end smoke run:
#   ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/patchify_train_eval.sh
#
# Note: `mamba`/`bimamba` need a CUDA GPU, and `mamba2` + `hpp_ultra_tiny` (48) is invalid
# (head-dim divisibility) — use `mamba1` there. Same caveats as the baseline.
#
set -euo pipefail

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

# Experiment shape: same three sequence models, but on the PatchTrunk acoustic model.
# EXTRA_CONFIGS adds the `patchify` named config (trunk='patch'); RUN_TAG_PREFIX prefixes the
# run names / logdirs with `patch_` so these never collide with the CNNTrunk runs.
VARIANTS="${VARIANTS:-lstm mamba bimamba}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-patchify}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-patch_}"

hppnet_setup
run_ablation
