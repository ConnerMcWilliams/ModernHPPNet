#!/usr/bin/env bash
#
# runpod_train_eval.sh — the baseline sequence-model ablation for HPPNet+.
#
# On a fresh RunPod CUDA box this does a one-shot setup + train + evaluate of the three
# sequence-model variants (lstm / mamba / bimamba) on the default CNNTrunk acoustic model.
# Each variant becomes one wandb run in $WANDB_PROJECT, grouped by $EXPERIMENT, so the three
# show up side-by-side for comparison (training curves + final MAESTRO-test metrics).
#
# All of the environment setup + safeguards + the train/eval loop live in scripts/lib/common.sh
# (sourced below) so they are shared with — and never re-fixed per — other experiment scripts.
# This script only picks the experiment shape. To add a *different* experiment, copy this file,
# don't edit it (see CLAUDE.md: "Creating a new experiment").
#
# Usage:
#   export WANDB_API_KEY=xxxxxxxx           # from https://wandb.ai/authorize
#   bash scripts/runpod_train_eval.sh
#
# Everything is overridable via environment variables (VARIANTS, SIZE_CONFIG, ITERATIONS, …),
# e.g. a cheap end-to-end smoke run:
#   ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/runpod_train_eval.sh
#
set -euo pipefail

# shellcheck source=lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

# Experiment shape: the three sequence models on the default CNNTrunk. (VARIANTS/SIZE_CONFIG
# etc. all fall back to common.sh defaults and stay env-overridable.)
hppnet_setup
run_ablation
