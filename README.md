# PyTorch Implementation of HPPNet Piano Transcription Model


This is a [PyTorch](https://pytorch.org/) implementation of [HPPNet](https://arxiv.org/abs/2208.14339) model, using the [Maestro dataset v3](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.



## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Downloading Dataset

To download the Maestro dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

Every stage — download, extraction, and FLAC conversion — runs in parallel across your CPU cores. The download uses `aria2c` when available, otherwise it falls back to parallel ranged `curl` requests; extraction prefers `unzip` and falls back to `bsdtar`. If `ffmpeg` is not installed the FLAC conversion step is skipped. The script's behavior can be tuned with environment variables:

- `JOBS` — number of parallel workers (default: `nproc`)
- `URL` — dataset URL (default: MAESTRO v3.0.0)
- `ZIP` — local zip filename (default: basename of `URL`)

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

**nnAudio / numpy caveat.** `nnAudio 0.2.6` uses numpy's removed `np.float` alias, so on
`numpy >= 1.24` (what a fresh `pip install -r requirements.txt` resolves to) `from hppnet import *`
— run by both `train.py` and `evaluate.py` — fails at import with
`AttributeError: module 'numpy' has no attribute 'float'`. Fix it by either installing `numpy < 1.24`
**or** rewriting the bare `np.float` alias to `float` in the installed nnAudio package:

```bash
# option 2: patch the installed nnAudio in place (leaves np.float32/np.float64 intact)
NNAUDIO_DIR="$(python -c 'import nnAudio, os; print(os.path.dirname(nnAudio.__file__))')"
grep -rlZ --include='*.py' -E 'np\.float\b' "$NNAUDIO_DIR" | xargs -0 -r sed -i -E 's/np\.float\b/float/g'
```

`scripts/runpod_train_eval.sh` applies this patch automatically.

**setuptools / `pkg_resources` caveat.** `setuptools 82.0` (Feb 2026) removed the bundled
`pkg_resources`, which `sacred` (and `wandb`) still import at startup. On a fresh env — which now
resolves `setuptools >= 82` — both `train.py` and `evaluate.py` die immediately with
`ModuleNotFoundError: No module named 'pkg_resources'`. Fix it by capping setuptools:

```bash
pip install "setuptools<82"
```

`scripts/runpod_train_eval.sh` pins this automatically (and sets `PIP_CONSTRAINT` so later installs
don't undo it).

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Experiment tracking with Weights & Biases

Training and evaluation stream all metrics to [Weights & Biases](https://wandb.ai/) — wandb
replaces the project's old TensorBoard logging. Set your API key once, then train/evaluate as
usual:

```bash
export WANDB_API_KEY=xxxxxxxx                 # from https://wandb.ai/authorize
export WANDB_PROJECT=hppnet-mamba-ablation    # optional; this is the default
python train.py with hpp_tiny mamba logdir=runs/mamba
python evaluate.py runs/mamba/model-100000.pt MAESTRO test --wandb
```

`train.py` logs per-step losses, validation metrics, and piano-roll images; `evaluate.py --wandb`
logs the final test metrics (note/frame precision, recall, F1, …) plus a per-file table. Run
name/group/id come from the standard wandb env vars (`WANDB_NAME`, `WANDB_RUN_GROUP`,
`WANDB_RUN_ID`, `WANDB_RESUME`), so a training run and its evaluation can share one run id and land
on the same run. If `WANDB_API_KEY` is unset, training still proceeds and logs to a local disabled
run.

Independently of wandb, sacred keeps a file-based run log under `logdir`. The MongoDB observer that
older versions hard-wired is now off by default — set `HPPNET_MONGO=<host:port>` to re-enable it.

### Model & Sequence-Model Options

`train.py` exposes several [sacred](https://sacred.readthedocs.io/) *named configs* that can be
combined on the command line after `with`. They control two independent axes: the model **size**
and the **sequence model** used inside the frequency-grouped heads.

**Model size** — each of these collapses the network onto a single onset subnet carrying all four
heads (onset, frame, offset, velocity):

| Named config     | `model_size` |
|------------------|--------------|
| `hpp_base`       | 128          |
| `hpp_tiny`       | 64           |
| `hpp_ultra_tiny` | 48           |

**Sequence model (Mamba / BiMamba ablation)** — the frequency-grouped `BiLSTM` in each head can be
swapped for a [Mamba](https://github.com/state-spaces/mamba) state-space model. This is controlled by
two options:

- `seq_model`: `lstm` (baseline — reproduces the original HPPNet exactly) | `mamba` (causal) | `bimamba` (bidirectional)
- `mamba_impl`: `mamba1` | `mamba2` (only used when `seq_model != lstm`)

Convenience named configs bundle these:

| Named config | `seq_model` | `mamba_impl` |
|--------------|-------------|--------------|
| `mamba`      | `mamba`     | `mamba1`     |
| `bimamba`    | `bimamba`   | `mamba1`     |
| `mamba2`     | `mamba`     | `mamba2`     |
| `bimamba2`   | `bimamba`   | `mamba2`     |

Combine a size config with a sequence-model config, for example:

```bash
python train.py with hpp_tiny mamba          # model_size 64, causal Mamba
python train.py with hpp_base bimamba         # model_size 128, bidirectional Mamba
```

The default (no sequence-model config) is `seq_model=lstm`, i.e. the original HPPNet baseline.

**Trunk / acoustic model (patchify ablation)** — a third, independent axis swaps the harmonic
dilated-conv acoustic model (`CNNTrunk`) for an [Audio Mamba](https://arxiv.org/abs/2406.03344)
/ ViT-style **patch-embedding trunk** (`PatchTrunk`). It patchifies the CQT log-spectrogram (a
`Conv2d` with kernel/stride `(1, 4)` → one token per semitone, 88 freq tokens/frame, time
preserved), adds a learnable positional embedding, and runs a sequence model **over the frequency
axis** — testing whether the hand-designed harmonic prior can be learned instead. The downstream
frequency-grouped heads still model time, so the two axes stay cleanly factorized.

- `trunk`: `cnn` (baseline harmonic convs) | `patch` (patchify)
- `patch_trunk_depth`: number of stacked sequence blocks in the patch trunk (default 2)

The patch trunk reuses the **same `seq_model` axis** as the heads, so the patchify ablation runs on
all three sequence models. Bundled as the `patchify` named config, it composes with any size and
sequence-model config:

```bash
python train.py with hpp_tiny patchify            # patch trunk + LSTM
python train.py with hpp_tiny patchify mamba      # patch trunk + causal Mamba
python train.py with hpp_tiny patchify bimamba    # patch trunk + BiMamba
```

The default (no `patchify`) is `trunk=cnn`, i.e. the original harmonic acoustic model. As with the
sequence-model axis, `mamba` / `bimamba` here need a CUDA GPU (see below); `patchify` on its own runs
the LSTM path on CPU. The `mamba2` + `hpp_ultra_tiny` (48) divisibility caveat below applies to the
patch trunk's sequence model too — use `mamba1` there.

**Installing Mamba (CUDA only).** The Mamba options need a CUDA GPU with compute capability ≥ 7.0.
They are not installed by `requirements.txt` — install them separately. The easiest path uses
**prebuilt wheels** (no `nvcc` / CUDA toolkit, no compile). For the `torch2.4` / `cu12` / `cp310` /
`cxx11abiFALSE` stack (as pinned in `scripts/runpod_train_eval.sh`), grab the matching wheels from
the [state-spaces/mamba](https://github.com/state-spaces/mamba/releases) and
[Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) GitHub releases:

```bash
pip install "transformers<4.45"   # mamba-ssm 2.2.x needs the removed GreedySearchDecoderOnlyOutput
pip install \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  einops
```

Pick the wheels whose `cuXX` / `torchX.Y` / `cpXYZ` / `cxx11abi*` tags match your environment. If no
prebuilt wheel matches (your pod's torch / python / CUDA differ from the wheel ABI), fall back to
compiling against your PyTorch CUDA build, which does need a matching `nvcc`:

```bash
pip install --no-build-isolation causal-conv1d mamba-ssm einops
```

**`mamba2` caveat.** Mamba-2 requires `d_model * expand` to be divisible by its head dim (64). With
the default `expand=2` this holds for `model_size` 128 and 64, but **not** 48 — so use `mamba1`
(not `mamba2`) with `hpp_ultra_tiny`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash

python evaluate.py runs/transcriber/model-600000.pt MAPS test
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/transcriber/model-600000.pt MAESTRO test
```

### Full ablation on RunPod

`scripts/runpod_train_eval.sh` runs the whole comparison end-to-end on a fresh RunPod CUDA box: it
installs the system + Python dependencies (torch 2.4 + the prebuilt Mamba wheels), downloads and
prepares full MAESTRO v3, then trains **and** evaluates the three sequence-model variants — `lstm`
(the original HPPNet baseline), `mamba`, and `bimamba` — logging every result to one wandb project
so they line up side-by-side for comparison.

```bash
export WANDB_API_KEY=xxxxxxxx
bash scripts/runpod_train_eval.sh
```

Each variant becomes one wandb run (its training curves plus its final MAESTRO-test metrics),
grouped by `EXPERIMENT`. Key knobs, all env-overridable: `VARIANTS="lstm mamba bimamba"`,
`SIZE_CONFIG=hpp_tiny`, `TRUNK=cnn`, `ITERATIONS=100000`, `WANDB_PROJECT`, `EXPERIMENT`. Do a cheap
end-to-end check first before committing to the long run:

```bash
ITERATIONS=200 CHECKPOINT_INTERVAL=100 VALIDATION_INTERVAL=100 bash scripts/runpod_train_eval.sh
```

To run the **patchify** ablation instead, set `TRUNK=patch` — this sweeps the same three sequence
models (`lstm`/`mamba`/`bimamba`) with the patch-embedding trunk, and prefixes the run names/logdirs
with `patch_` so they never collide with the `cnn` runs:

```bash
TRUNK=patch bash scripts/runpod_train_eval.sh
```

## Acknowledgements

This project is based on the PyTorch implementation of Onsets and Frames model => https://github.com/jongwook/onsets-and-frames


## Citation

```
@inproceedings{Wei2022HPPNet,
  author       = {Weixing Wei and
                  Peilin Li and
                  Yi Yu and
                  Wei Li},
  title        = {HPPNet: Modeling the Harmonic Structure and Pitch Invariance in Piano
                  Transcription},
  booktitle    = {Proceedings of the 23rd International Society for Music Information
                  Retrieval Conference, {ISMIR} 2022, Bengaluru, India, December 4-8,
                  2022},
  pages        = {709--716},
  year         = {2022},
  url          = {https://archives.ismir.net/ismir2022/paper/000085.pdf},
}
```
