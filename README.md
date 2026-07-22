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

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

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

**Installing Mamba (CUDA only).** The Mamba options require compiling CUDA kernels, so they need a
GPU with compute capability ≥ 7.0 and a `nvcc` matching your PyTorch CUDA build. They are not
installed by `requirements.txt` — install them separately with:

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
