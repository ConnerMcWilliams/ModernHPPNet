#!/usr/bin/env bash
#
# runpod_train_eval.sh — one-shot setup + train + evaluate for the HPPNet+ Mamba ablation.
#
# On a fresh RunPod CUDA box this will:
#   1. install system deps (ffmpeg, unzip, git, wget)
#   2. create a reproducible Python 3.10 conda env with the known-good torch 2.4 + Mamba stack
#   3. log in to Weights & Biases
#   4. download + prepare the full MAESTRO v3 dataset
#   5. train AND evaluate each variant (lstm / mamba / bimamba), streaming everything to wandb
#
# Each variant becomes one wandb run in $WANDB_PROJECT, grouped by $EXPERIMENT, so the three
# show up side-by-side for comparison (training curves + final MAESTRO-test metrics).
#
# Usage:
#   export WANDB_API_KEY=xxxxxxxx           # from https://wandb.ai/authorize
#   bash scripts/runpod_train_eval.sh
#
# Everything below is overridable via environment variables, e.g. a cheap end-to-end smoke run:
#   ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/runpod_train_eval.sh
#
set -euo pipefail

# --------------------------------------------------------------------------------------------
# Configuration (override via env)
# --------------------------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${WANDB_API_KEY:?Set WANDB_API_KEY (get it from https://wandb.ai/authorize)}"
export WANDB_PROJECT="${WANDB_PROJECT:-hppnet-mamba-ablation}"
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY   # optional wandb team/user
EXPERIMENT="${EXPERIMENT:-$(date +%y%m%d-%H%M%S)}"  # shared group name for the runs

VARIANTS="${VARIANTS:-lstm mamba bimamba}"
SIZE_CONFIG="${SIZE_CONFIG:-hpp_tiny}"              # hpp_base | hpp_tiny | hpp_ultra_tiny
ITERATIONS="${ITERATIONS:-100000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-2000}"

CONDA_HOME="${CONDA_HOME:-/workspace/miniconda3}"   # persistent volume on RunPod
ENV_NAME="${ENV_NAME:-hppnet}"
SKIP_SETUP="${SKIP_SETUP:-0}"                       # set 1 to reuse an existing env
SKIP_DATA="${SKIP_DATA:-0}"                         # set 1 if MAESTRO already prepared

# Prebuilt Mamba wheels matching torch2.4 / cu12 / cp310 / cxx11abiFALSE (no compile needed).
CAUSAL_CONV1D_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA_SSM_WHL="https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

cd "$REPO_DIR"

# --------------------------------------------------------------------------------------------
# 1. System dependencies
# --------------------------------------------------------------------------------------------
if [ "$SKIP_SETUP" != "1" ]; then
  echo "==> Installing system packages (ffmpeg, unzip, git, wget)"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg unzip git wget
  else
    echo "[warn] apt-get not found; ensure ffmpeg, unzip, git, wget are installed" >&2
  fi
fi

# --------------------------------------------------------------------------------------------
# 2. Python env (Miniconda + torch 2.4 + Mamba). Idempotent: reused across pod restarts.
# --------------------------------------------------------------------------------------------
if [ ! -x "$CONDA_HOME/bin/conda" ] && [ "$SKIP_SETUP" != "1" ]; then
  echo "==> Bootstrapping Miniconda into $CONDA_HOME"
  mkdir -p "$(dirname "$CONDA_HOME")"
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_HOME"
  rm -f /tmp/miniconda.sh
fi

# conda's own scripts trip over `set -u`; relax it just around sourcing/activation.
set +u
# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

# Recent Miniconda gates the Anaconda default channels behind a Terms-of-Service
# prompt that aborts non-interactive `conda create`. Accept it up front (no-op if
# already accepted, or if this conda build predates the `tos` subcommand).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Creating conda env '$ENV_NAME' (python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
set -u
PYTHON="$(command -v python)"

# setuptools 82.0 (Feb 2026) removed the bundled `pkg_resources`, which sacred still imports at
# startup (and so do wandb / some other deps) -> "ModuleNotFoundError: No module named
# 'pkg_resources'" the instant train.py/evaluate.py run. Fresh envs now resolve setuptools>=82,
# so cap it and put `pkg_resources` back. Done unconditionally (even with SKIP_SETUP=1) so an
# env created by an earlier run gets repaired too. The PIP_CONSTRAINT file makes every later
# `pip install` — including PEP 517 build-isolation envs (e.g. the git+mir_eval build) — honor
# the cap, so nothing pulls setuptools>=82 back in.
export PIP_CONSTRAINT="${PIP_CONSTRAINT:-/tmp/hppnet-pip-constraints.txt}"
printf 'setuptools<82\n' > "$PIP_CONSTRAINT"
echo "==> Pinning setuptools<82 (restores pkg_resources, removed in setuptools 82.0)"
pip install "setuptools<82" wheel

if [ "$SKIP_SETUP" != "1" ]; then
  echo "==> Installing Python dependencies (torch 2.4.1 + Mamba stack)"
  pip install --upgrade pip
  # Pinned, known-good stack. torch first so the Mamba wheels match its ABI.
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install "numpy<2"            # nnAudio 0.2.6 needs a bare-np.float patch (removed in numpy>=1.24, applied below)
  pip install "transformers<4.45"  # mamba-ssm 2.2.x imports the removed GreedySearchDecoderOnlyOutput
  # Prebuilt Mamba kernels (no nvcc/build needed). If your pod's torch/python differ from the
  # wheel ABI above, replace the next line with the compile path:
  #   pip install --no-build-isolation causal-conv1d mamba-ssm einops
  pip install "$CAUSAL_CONV1D_WHL" "$MAMBA_SSM_WHL" einops
  pip install -r requirements.txt

  # nnAudio 0.2.6 (installed above) still uses the bare `np.float` alias, removed in numpy>=1.24
  # which the numpy<2 pin resolves to. hppnet imports nnAudio at module load, so rewrite the
  # alias to `float` in place; this is what actually makes `from hppnet import *` work. The word
  # boundary leaves np.float32/np.float64 intact, and the grep|xargs form is a no-op once patched.
  NNAUDIO_DIR="$("$PYTHON" -c 'import nnAudio, os; print(os.path.dirname(nnAudio.__file__))')"
  { grep -rlZ --include='*.py' -E 'np\.float\b' "$NNAUDIO_DIR" || true; } | xargs -0 -r sed -i -E 's/np\.float\b/float/g'
fi

# --------------------------------------------------------------------------------------------
# 3. Weights & Biases auth
# --------------------------------------------------------------------------------------------
echo "==> Logging in to Weights & Biases"
wandb login "$WANDB_API_KEY"

# --------------------------------------------------------------------------------------------
# 4. Dataset (full MAESTRO v3 -> 16 kHz mono FLAC)
# --------------------------------------------------------------------------------------------
maestro_ready() {
  # "Ready" means the metadata CSV is present AND at least one converted FLAC exists.
  # The dataset loader keeps only rows whose 16 kHz .flac is on disk (the original
  # 44.1 kHz WAVs are unusable), so a CSV-only tree — extracted but not yet converted —
  # sails past a CSV check and then dies at training with num_samples=0. prepare_maestro.sh
  # is resumable, so re-running here just fills in the missing FLACs (no re-download).
  [ -f "data/maestro-v3.0.0/maestro-v3.0.0.csv" ] \
    && [ -n "$(find data/maestro-v3.0.0 -name '*.flac' -print -quit 2>/dev/null)" ]
}

if [ "$SKIP_DATA" != "1" ] && ! maestro_ready; then
  echo "==> Preparing MAESTRO v3 (download/extract are skipped if already present; needs ~200 GB scratch)"
  ( cd data && bash ./prepare_maestro.sh )
  if ! maestro_ready; then
    echo "[error] MAESTRO still has no FLAC files after prepare_maestro.sh — check disk space" >&2
    echo "        (df -h /workspace) and ffmpeg; the dataset loader needs 16 kHz FLACs." >&2
    exit 1
  fi
else
  echo "==> Skipping dataset prep (already present or SKIP_DATA=1)"
fi

# --------------------------------------------------------------------------------------------
# 5. Train + evaluate each variant, streaming to wandb
# --------------------------------------------------------------------------------------------
export WANDB_RUN_GROUP="$EXPERIMENT"
export WANDB_RESUME="allow"

seq_config() {
  case "$1" in
    lstm)     echo "" ;;             # default seq_model=lstm — original HPPNet baseline
    mamba)    echo "mamba" ;;
    bimamba)  echo "bimamba" ;;
    mamba2)   echo "mamba2" ;;
    bimamba2) echo "bimamba2" ;;
    *) echo "[error] unknown variant: $1" >&2; exit 1 ;;
  esac
}

for v in $VARIANTS; do
  seqcfg="$(seq_config "$v")"
  logdir="runs/${EXPERIMENT}/${v}"
  export WANDB_NAME="$v"
  export WANDB_RUN_ID="${EXPERIMENT}-${v}"

  echo "=========================================================================="
  echo "==> [$v] training  (size=$SIZE_CONFIG, iters=$ITERATIONS, logdir=$logdir)"
  echo "=========================================================================="
  WANDB_JOB_TYPE=train "$PYTHON" train.py with $SIZE_CONFIG $seqcfg \
    logdir="$logdir" \
    iterations="$ITERATIONS" \
    checkpoint_interval="$CHECKPOINT_INTERVAL" \
    validation_interval="$VALIDATION_INTERVAL"

  # Evaluate the final checkpoint (fall back to the newest one if ITERATIONS isn't a
  # multiple of CHECKPOINT_INTERVAL).
  ckpt="$logdir/model-${ITERATIONS}.pt"
  if [ ! -f "$ckpt" ]; then
    ckpt="$(ls -1v "$logdir"/model-*.pt 2>/dev/null | tail -n1 || true)"
  fi
  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    echo "[error] no checkpoint found in $logdir" >&2
    exit 1
  fi

  echo "==> [$v] evaluating on MAESTRO test ($ckpt)"
  WANDB_JOB_TYPE=eval "$PYTHON" evaluate.py "$ckpt" MAESTRO test --device cuda --wandb
done

echo "==> All done. Compare runs in wandb project '$WANDB_PROJECT', group '$EXPERIMENT'."
