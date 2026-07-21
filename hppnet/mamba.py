"""Mamba / bidirectional-Mamba sequence blocks for the HPPNet ablation.

These are drop-in replacements for the ``BiLSTM`` used inside
``FreqGroupLSTM`` (see ``hppnet/nets.py``). They keep the exact same
tensor interface -- input ``[N, T, in_features]`` -> output
``[N, T, out_features]`` -- so the frequency-grouping wrapper and the
downstream ``Linear(lstm_size, channel_out)`` are reused unchanged. Only
the *sequence model over time* is swapped, isolating its effect.

Backend: the official ``mamba-ssm`` CUDA kernels. The import is lazy (it
happens inside ``_make_mamba`` only when a Mamba variant is actually
selected) so that the default LSTM path stays importable on CPU-only
machines. Training/evaluating the Mamba ablation requires a CUDA GPU and:

    pip install --no-build-isolation causal-conv1d mamba-ssm
"""

import torch
from torch import nn


def _make_mamba(d_model, impl='mamba1'):
    """Build a single Mamba block with ``d_model`` in == out features.

    ``impl='mamba1'`` uses the original selective-SSM ``Mamba`` (no
    dimensional constraints). ``impl='mamba2'`` uses ``Mamba2``, whose
    faster kernels require ``d_model * expand`` to be divisible by
    ``headdim`` (default 64): OK for ``d_model in {128, 64, 32}`` with
    ``expand=2``, but e.g. ``d_model=48`` (``hpp_ultra_tiny`` bidir would
    even be 24) is not -- use ``mamba1`` there or pass a smaller headdim.
    """
    if impl == 'mamba2':
        from mamba_ssm import Mamba2  # CUDA kernels; lazy import
        return Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2)
    elif impl == 'mamba1':
        from mamba_ssm import Mamba  # CUDA kernels; lazy import
        return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    else:
        raise ValueError(f'unknown mamba_impl: {impl}')


class MambaSeq(nn.Module):
    """Causal (unidirectional) Mamba over the time axis.

    ``[N, T, in_features]`` -> ``[N, T, out_features]``.
    """

    def __init__(self, in_features, out_features, impl='mamba1'):
        super().__init__()
        self.in_proj = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.mamba = _make_mamba(out_features, impl)

    def forward(self, x):
        # [N x T x in] => [N x T x out]
        x = self.in_proj(x).contiguous()
        # pre-norm residual
        return x + self.mamba(self.norm(x))


class BiMambaSeq(nn.Module):
    """Bidirectional Mamba: a forward and a time-reversed stream, concatenated.

    Structurally mirrors ``BiLSTM(in_features, out_features // 2)`` whose two
    directions concatenate to ``out_features``. ``[N, T, in_features]`` ->
    ``[N, T, out_features]``.
    """

    def __init__(self, in_features, out_features, impl='mamba1'):
        super().__init__()
        assert out_features % 2 == 0, 'out_features must be even for BiMamba'
        half = out_features // 2
        self.in_proj = nn.Linear(in_features, half)
        self.norm = nn.LayerNorm(half)
        self.fwd = _make_mamba(half, impl)
        self.bwd = _make_mamba(half, impl)

    def forward(self, x):
        # [N x T x in] => [N x T x half]
        x = self.in_proj(x).contiguous()
        xn = self.norm(x)
        # forward direction
        f = x + self.fwd(xn)
        # backward direction: flip time, run, flip back
        b = x + torch.flip(self.bwd(torch.flip(xn, [1])), [1])
        # => [N x T x out]
        return torch.cat([f, b], dim=-1)
