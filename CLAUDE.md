# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this project is

**ModernHPPNet** is a PyTorch implementation of **HPPNet** ([arXiv:2208.14339](https://arxiv.org/abs/2208.14339)),
a piano transcription model — it converts raw piano audio into note events (MIDI). It builds on the
original HPPNet PyTorch code and adds three things over the baseline:

1. A **Mamba / BiMamba sequence-model ablation** — the frequency-grouped `BiLSTM` inside each output
   head can be swapped for a (bi)directional [Mamba](https://github.com/state-spaces/mamba) SSM.
2. A **patchify-trunk ablation** — the harmonic dilated-conv acoustic model (`CNNTrunk`) can be
   swapped for an AuM/ViT-style patch-embedding trunk (`PatchTrunk`) via `trunk='patch'`.
3. **Weights & Biases** experiment tracking (replacing the project's old TensorBoard logging).

Training uses the **Maestro v3** dataset; testing typically uses the Disklavier portion of the **MAPS**
database. The pipeline: audio → CQT log-spectrogram → CNN/patch trunk (harmonic dilated convs or
patch embedding) →
frequency-grouped sequence heads (onset / frame / offset / velocity) → note decoding → MIDI.

This is a research/experimentation codebase, not a packaged library — there are no tests, no CI, and
no `setup.py`. Entry points are top-level scripts run directly with `python`.

## Repository layout

```
train.py            # Training entry point (sacred Experiment). Defines all configs & named configs.
evaluate.py         # Evaluation entry point + reusable evaluate() used by train.py's validation loop.
transcribe.py       # Run a trained model on arbitrary audio -> MIDI (no ground truth needed).
requirements.txt    # Python deps. Read the inline comments — several pins matter (see Gotchas).
README.md           # User-facing setup/usage docs; keep it in sync with behavior changes.

hppnet/             # The model package. `from hppnet import *` re-exports the public API (__init__.py).
  constants.py      #   Audio/MIDI constants: SAMPLE_RATE=16000, HOP_LENGTH, MIN/MAX_MIDI (21..108), etc.
  transcriber.py    #   HPPNet top-level module: forward pass, run_on_batch (losses), SubNet/Head.
  nets.py           #   CNNTrunk (harmonic dilated convs), PatchTrunk (AuM/ViT patchify trunk), FreqGroupLSTM (freq-grouped seq wrapper).
  lstm.py           #   BiLSTM — the baseline sequence model (chunked inference for long audio).
  mamba.py          #   MambaSeq / BiMambaSeq — drop-in replacements for BiLSTM (ablation).
  dataset.py        #   MAESTRO / MAPS PianoRollAudioDataset; caches preprocessed data as HDF5.
  decoding.py       #   extract_notes / notes_to_frames — piano-roll <-> note-event conversion.
  midi.py           #   parse_midi (MIDI -> note rows, handles sustain pedal) / save_midi.
  utils.py          #   summary(), piano-roll image saving, source-file snapshotting, cycle().
  layers.py         #   WaveformToLogSpecgram — alt spectrogram front-end (currently unused by HPPNet).
  mel.py            #   Mel spectrogram helper (legacy; import commented out in __init__.py).

scripts/
  lib/common.sh          # SOURCED shared library: all env setup + safeguards + the train/eval loop.
  runpod_train_eval.sh   # Experiment: baseline lstm/mamba/bimamba on CNNTrunk (sources lib/common.sh).
  patchify_train_eval.sh # Experiment: same 3 seq models on PatchTrunk (sources lib/common.sh).
data/
  prepare_maestro.sh     # Downloads Maestro v3, unzips, transcodes wav -> 16 kHz mono FLAC.
  .keep                  # Datasets are gitignored; they live here at runtime.
```

## Core architecture (how the model is wired)

- **`HPPNet`** (`transcriber.py`) owns a `to_cqt` nnAudio CQT front-end and one or two **`SubNet`s**.
  Which subnets exist is driven by `config['SUBNETS_TO_TRAIN']` (`'onset_subnet'`, `'frame_subnet'`).
- Each **`SubNet`** = a shared **trunk** ("acoustic model") + a `nn.ModuleDict` of **`Head`s** (one per
  head name). Head names come from `config['onset_subnet_heads']` / `config['frame_subnet_heads']` and
  correspond to prediction keys: `onset`, `frame`, `offset`, `velocity`.
- The trunk is one of two, selected by `config.get('trunk', 'cnn')`:
  - **`CNNTrunk`** (`nets.py`, default) is the harmonic feature extractor: plain conv blocks → a
    `HarmonicDilatedConv` (8 parallel dilations targeting harmonic spacing on the log-freq axis) → more
    conv blocks. Uses `InstanceNorm2d`.
  - **`PatchTrunk`** (`nets.py`, `trunk='patch'`) is the patchify ablation: an AuM/ViT-style patch
    embedding (`Conv2d` kernel/stride `(1,4)` → 88 freq tokens/frame) + learnable positional embedding,
    then a sequence model **over the frequency axis** (reusing the same `seq_model` lstm/mamba/bimamba).
  - **Any trunk must honor the same output contract:** input `[B x 1 x T x 352]` → output
    `[B x model_size x T x 88]`, with `T` preserved exactly (the onset subnet reshapes head output
    straight to `[B x T x 88]` in `run_on_batch` — no interpolation) and freq == 88 (the frame subnet's
    `F.interpolate(..., size=src_size[-2:])` with `src_size[-1]=88` depends on it).
- **`Head`** wraps a **`FreqGroupLSTM`**, which reshapes `[B x C x T x freq]` so the sequence model
  runs **over the time axis independently per frequency bin**, then a `Linear` + `sigmoid` per bin.
- **The ablation seam is `FreqGroupLSTM.__init__`**: `seq_model` selects `BiLSTM` (`'lstm'`),
  `MambaSeq` (`'mamba'`), or `BiMambaSeq` (`'bimamba'`). The Mamba classes are **drop-in** — identical
  `[N, T, in] -> [N, T, out]` interface — so everything downstream is reused unchanged. **Preserve this
  interface** when touching sequence models.
- **Losses** live in `HPPNet.run_on_batch`: weighted BCE for onset/frame/offset, a masked MSE for
  velocity. Each subnet is optimized by its **own Adam optimizer** (see the training loop) — this is
  intentional, not a refactor target.

### Label & tensor conventions

- Piano roll has **88 keys** (`MIN_MIDI=21`..`MAX_MIDI=108`). Time axis is frames at `HOP_LENGTH`
  (20 ms hops @ 16 kHz).
- Dataset label encoding (`dataset.py`): `3 = onset`, `2 = frame-after-onset`, `1 = offset`,
  `0 = silence`. Derived masks: `onset = (label==3)`, `offset = (label==1)`, `frame = (label>1)`.
- Predictions/labels are shaped `[B x T x 88]`. Head outputs are clipped to `[1e-7, 1-1e-7]`.

## Common commands

Always run from the repo root.

```bash
# Install deps (read the Gotchas section first — there's an nnAudio/numpy caveat)
pip install -r requirements.txt

# Train the LSTM baseline (original HPPNet)
python train.py

# Train with named configs after `with` (sacred). Combine a SIZE with a SEQUENCE-MODEL config:
python train.py with hpp_tiny mamba logdir=runs/mamba iterations=100000
python train.py with hpp_base bimamba

# Override any config value inline
python train.py with logdir=runs/model iterations=1000000 batch_size=2

# Evaluate a checkpoint (note/frame precision, recall, F1, ...)
python evaluate.py runs/model/model-100000.pt MAESTRO test
python evaluate.py runs/model/model-600000.pt MAPS test --save-path output/

# Transcribe arbitrary audio -> MIDI + piano-roll images
python transcribe.py path/to/audio.flac --model_file runs/model/model-100000.pt --save-path out/

# Full 3-variant (lstm/mamba/bimamba) ablation on a fresh RunPod CUDA box
export WANDB_API_KEY=xxxxxxxx
bash scripts/runpod_train_eval.sh              # baseline: the 3 seq models on CNNTrunk
bash scripts/patchify_train_eval.sh            # patchify ablation: the 3 seq models on PatchTrunk
# Cheap end-to-end smoke test first (works for either experiment script):
ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/runpod_train_eval.sh
```

## Configuration model (sacred)

`train.py` is a [sacred](https://sacred.readthedocs.io/) `Experiment`. Configuration is code, set via
`@ex.config` functions, and overridden on the CLI after the keyword **`with`**. **Named configs**
(`@ex.named_config`) are toggled by naming them after `with`. Three independent axes:

| Axis | Named configs | Effect |
|------|---------------|--------|
| **Model size** | `hpp_base` (128), `hpp_tiny` (64), `hpp_ultra_tiny` (48) | Set `model_size`; collapse to a single `onset_subnet` carrying all four heads. |
| **Sequence model** | `mamba`, `bimamba`, `mamba2`, `bimamba2` | Set `seq_model` (`mamba`/`bimamba`) and `mamba_impl` (`mamba1`/`mamba2`). |
| **Trunk (acoustic model)** | `patchify` | Set `trunk='patch'` (`PatchTrunk`) instead of the default `'cnn'` (`CNNTrunk`). Composes with any size + sequence-model config; the patch trunk reuses the same `seq_model` axis. |

Default (no sequence-model config) is `seq_model='lstm'` and (no `patchify`) `trunk='cnn'` — the exact
original HPPNet baseline. When
adding a config knob, add it to the appropriate `@ex.config` function and thread it through
`HPPNet.__init__` via the `config` dict (use `config.get('key', default)` to stay backward-compatible
with older saved models/configs, as `seq_model`/`mamba_impl` do).

Checkpoints are saved as `model-<iter>.pt` under `logdir` (default: a timestamped `runs/transcriber-*`).
They are **whole-module `torch.save`** (not just `state_dict`), so loading requires the class
definitions to remain importable/compatible.

## Experiment tracking (Weights & Biases)

- Training and evaluation stream metrics to **wandb** (`WANDB_API_KEY` from https://wandb.ai/authorize).
  Default project: `hppnet-mamba-ablation` (override with `WANDB_PROJECT`).
- Run name/group/id/resume come from standard env vars (`WANDB_NAME`, `WANDB_RUN_GROUP`,
  `WANDB_RUN_ID`, `WANDB_RESUME`, `WANDB_JOB_TYPE`) so a training run and its evaluation can share one
  run id and land on the same run. The RunPod experiment scripts (via `lib/common.sh`'s `run_ablation`)
  wire these up per variant.
- **wandb is best-effort**: if `WANDB_API_KEY` is unset, training/eval still proceed against a local
  disabled run — never let a wandb hiccup break the actual work. Preserve this fallback behavior.
- Sacred also keeps a file-based run log under `logdir`. The old MongoDB observer is **off by default**;
  set `HPPNET_MONGO=<host:port>` to re-enable it.

## RunPod experiment scripts (safeguards & adding a new one)

The RunPod scripts are structured so the **hard-won environment fixes live in exactly one place** and
are never re-discovered per experiment. If you find yourself re-fixing any of the safeguards below,
something is wrong — they should already be inherited.

**Structure — one shared library, one thin script per experiment:**

- **`scripts/lib/common.sh`** is **sourced, never executed**. It owns *everything* reusable: all
  environment setup (system deps, conda env, the pinned torch+Mamba stack, the nnAudio patch, wandb
  login, MAESTRO prep) **and** the `run_ablation` train→eval loop. All the safeguards below live here.
- **Each experiment is its own thin script** (`runpod_train_eval.sh` = baseline CNNTrunk;
  `patchify_train_eval.sh` = PatchTrunk). A script sources `lib/common.sh`, sets a few knobs
  (`VARIANTS`, `EXTRA_CONFIGS` = extra sacred named configs, `RUN_TAG_PREFIX` = run-name/logdir
  prefix), then calls `hppnet_setup` and `run_ablation`. That's the whole script.
- **Never bake a new ablation axis into an existing experiment's script** (no growing `TRUNK=`-style
  switches inside `runpod_train_eval.sh`). New experiment ⇒ **new script**. It keeps each script a
  self-documenting record of one experiment and stops the shared setup from drifting between copies.

**Creating a new experiment** — copy `patchify_train_eval.sh` to `scripts/<name>_train_eval.sh` and
change only the knobs:

1. If it needs a new config, add a `@ex.named_config` in `train.py` (thread it through
   `HPPNet.__init__` via `config.get('key', default)` — keep it optional), and update `README.md`.
2. In the new script, set `EXTRA_CONFIGS="<your named config>"` and a unique `RUN_TAG_PREFIX="<tag>_"`
   so its wandb runs / logdirs never collide with other experiments' arms.
3. Set `VARIANTS`/`SIZE_CONFIG` if the sweep differs from the default `lstm mamba bimamba` / `hpp_tiny`.
4. Leave `hppnet_setup` + `run_ablation` as-is — do **not** copy setup logic into the script.
5. Smoke-test end-to-end before the long run:
   `ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/<name>_train_eval.sh`.

**Safeguards baked into `scripts/lib/common.sh`** (each one is a past fix — leave them in place):

- **Run with `bash`, not `sh`/`dash`.** The scripts use `set -euo pipefail`; dash aborts with
  `set: Illegal option -o pipefail`. Every invocation and doc uses `bash scripts/...`.
- **`set +u` around conda.** conda's own `profile.d` scripts reference unset vars, so `set -u` is
  relaxed only around `source .../conda.sh` + `conda activate`, then restored.
- **conda ToS auto-accept.** Recent Miniconda gates the default channels behind a Terms-of-Service
  prompt that aborts non-interactive `conda create`; `conda tos accept` for the `main`/`r` channels
  runs up front (`|| true`, so it's a no-op on older conda).
- **`setuptools<82` + `PIP_CONSTRAINT`.** setuptools 82.0 removed bundled `pkg_resources`, which
  `sacred`/`wandb` import at startup. The cap is applied **unconditionally** (even `SKIP_SETUP=1`, to
  repair older envs) and via a `PIP_CONSTRAINT` file so build-isolation installs honor it too.
- **`numpy<2` + the nnAudio `np.float` patch.** nnAudio 0.2.6 uses the removed `np.float` alias;
  a `grep -rlZ … || true | xargs -0 -r sed` rewrites it to `float` in the installed package. The
  form is **idempotent under `pipefail`** (no match ⇒ still exit 0) and leaves `np.float32/64` intact.
- **`transformers<4.45`.** mamba-ssm 2.2.x imports a symbol removed in newer transformers.
- **Prebuilt Mamba wheels pinned to the torch2.4/cu12/cp310 ABI.** Avoids an nvcc compile; if the
  pod's torch/python differ, switch to the documented `--no-build-isolation` compile path.
- **MAESTRO readiness self-heal (`maestro_ready`).** A CSV-only tree (extracted but not yet
  transcoded) passes a naïve CSV check and then dies at training with `num_samples=0`. Readiness
  requires the CSV **and** at least one `.flac`; if missing, it re-runs the resumable
  `prepare_maestro.sh` (no re-download) and fails loudly if still empty.
- **Checkpoint fallback.** If `model-<ITERATIONS>.pt` is absent (iterations not a multiple of the
  checkpoint interval), eval falls back to the newest `model-*.pt`.
- **Idempotent / resumable.** `SKIP_SETUP=1` reuses an env, `SKIP_DATA=1` skips prep, and the conda
  env is reused across pod restarts.

## Datasets

- `data/prepare_maestro.sh` downloads Maestro v3 (~101 GB), unzips, and transcodes to 16 kHz mono FLAC
  (needs `ffmpeg` and ~200 GB scratch). Data dirs are gitignored (`data*`, `runs`, `wandb`, `output*`).
- `PianoRollAudioDataset` **caches preprocessed audio+labels as HDF5** in a sibling `h5_hop320/` dir on
  first load, and caches MIDI→`.tsv` note tables next to the MIDI. Delete these caches if you change
  the preprocessing (hop length, label encoding, etc.), or stale data will be silently reused.
- The project is resource-heavy: **≥32 GB RAM and ≥8 GB GPU** recommended. `train.py` auto-halves
  `batch_size`/`sequence_length` on GPUs with <10 GB.

## Gotchas & conventions

- **nnAudio / numpy `np.float`**: `nnAudio==0.2.6` uses numpy's removed `np.float` alias, so on
  `numpy >= 1.24` `from hppnet import *` (run by `train.py`/`evaluate.py`) fails at import. Fix by
  installing `numpy < 1.24` **or** rewriting the bare `np.float`→`float` in the installed nnAudio
  package. `scripts/lib/common.sh` applies this patch automatically. This is the single most
  common setup failure — check it first when imports break.
- **Mamba is CUDA-only and not in `requirements.txt`**. It needs a GPU with compute capability ≥ 7.0,
  plus `causal-conv1d`, `mamba-ssm`, `einops` (and `transformers<4.45` for mamba-ssm 2.2.x). Prefer the
  prebuilt wheels (see README). The imports in `mamba.py` are **lazy** on purpose so the default LSTM
  path stays importable on CPU-only machines — keep them lazy.
- **`mamba2` + `hpp_ultra_tiny` is invalid**: Mamba-2 needs `d_model * expand` divisible by head dim
  (64). Holds for sizes 128/64 with `expand=2`, **not** 48. Use `mamba1` with `hpp_ultra_tiny`.
- **Dependency pins are load-bearing** — read the comments in `requirements.txt` and
  `scripts/lib/common.sh` before bumping versions (`sacred>=0.8.7`, `nnAudio==0.2.6`, torch 2.4 stack,
  `transformers<4.45`, `setuptools<82`).
- **`setuptools<82` / `pkg_resources`**: setuptools 82.0 (Feb 2026) removed the bundled
  `pkg_resources`, which `sacred` (imported at the top of `train.py`/`evaluate.py`) and `wandb` still
  `import` — so a fresh env (now resolving `setuptools>=82`) dies immediately with
  `ModuleNotFoundError: No module named 'pkg_resources'`. `scripts/lib/common.sh` pins `setuptools<82`
  and sets `PIP_CONSTRAINT` so nothing re-upgrades it. Keep that pin.
- The codebase carries a lot of **commented-out experimental code** (alternate front-ends, old metric
  blocks, mel path). This is deliberate history, not dead code to clean up unless asked.
- Some source comments are in Chinese (e.g. `.vscode/launch.json`, a note in `layers.py`). Leave them.

## Workflow for changes

- **Match the surrounding style** — this code favors explicit tensor-shape comments (`# => [B x T x 88]`),
  minimal abstraction, and direct scripts over frameworks. Don't introduce large refactors, type
  systems, or a test harness unless asked.
- When you change model architecture, config options, or run commands, **update `README.md`** to match —
  it's the user-facing contract and is currently kept carefully in sync.
- There are **no automated tests or linters** configured. Validate changes by running a short training
  smoke test (small `iterations`, `device=cpu` for the LSTM path if no GPU) and confirming imports work.
- Preserve backward compatibility with existing saved `.pt` checkpoints where reasonable (whole-module
  pickles), and keep new config keys optional via `config.get(...)`.

## Git conventions

- Develop on the designated feature branch; commit with clear, descriptive messages and push with
  `git push -u origin <branch>`.
- Recent history uses prefixes like `no-mistakes(review):` / `no-mistakes(lint):` for follow-up fixes;
  otherwise short imperative subjects. Match whatever the current branch is doing.
- Do **not** open a pull request unless explicitly asked.
</content>
</invoke>
