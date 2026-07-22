from lib2to3.pgen2.token import NT_OFFSET
from math import sin
from multiprocessing import pool
import torch

import torch.nn as nn
import torch.nn.functional as F
import nnAudio
import torchaudio

import torchvision
import matplotlib.pyplot as plt
import os

from .constants import *
from .lstm import BiLSTM


class FreqGroupLSTM(nn.Module):
    def __init__(self, channel_in, channel_out, lstm_size,
                 seq_model='lstm', mamba_impl='mamba1') -> None:
        super().__init__()

        self.channel_out = channel_out

        # Sequence model over the time axis (shared across frequency groups).
        # 'lstm' reproduces the original HPPNet baseline exactly; 'mamba' /
        # 'bimamba' swap in a (bi)directional Mamba SSM for the ablation.
        if seq_model == 'lstm':
            self.seq = BiLSTM(channel_in, lstm_size//2)
        elif seq_model == 'mamba':
            from .mamba import MambaSeq
            self.seq = MambaSeq(channel_in, lstm_size, impl=mamba_impl)
        elif seq_model == 'bimamba':
            from .mamba import BiMambaSeq
            self.seq = BiMambaSeq(channel_in, lstm_size, impl=mamba_impl)
        else:
            raise ValueError(f'unknown seq_model: {seq_model}')
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x):
        # inputs: [b x c_in x T x freq]
        # outputs: [b x c_out x T x freq]

        b, c_in, t, n_freq = x.size() 

        # => [b x freq x T x c_in] 
        x = torch.permute(x, [0, 3, 2, 1])

        # => [(b*freq) x T x c_in]
        x = x.reshape([b*n_freq, t, c_in])
        # => [(b*freq) x T x lstm_size]
        x = self.seq(x)
        # => [(b*freq) x T x c_out]
        x = self.linear(x)
        # => [b x freq x T x c_out]
        x = x.reshape([b, n_freq, t, self.channel_out])
        # => [b x c_out x T x freq]
        x = torch.permute(x, [0, 3, 2, 1])
        x = torch.sigmoid(x)
        return x

class PatchTrunk(nn.Module):
    """AuM/ViT-style patch-embedding trunk -- a drop-in replacement for ``CNNTrunk``.

    Instead of the hand-designed harmonic dilated convolutions, this patchifies the
    CQT log-spectrogram (like Audio Mamba treats a spectrogram as an image) and runs
    a sequence model over the *frequency* axis to mix information across pitches. The
    downstream ``FreqGroupLSTM`` heads still model the time axis, so the two axes are
    cleanly factorized.

    Same tensor contract as ``CNNTrunk``: input ``[B x 1 x T x 352]`` -> output
    ``[B x embedding x T x 88]``, with ``T`` preserved exactly and freq == 88.

    ``patch_f = BINS_PER_SEMITONE`` (4) maps the 352 CQT bins to exactly 88 tokens
    (one per semitone / piano key), and ``patch_t = 1`` preserves the time dimension,
    so no un-patchify or interpolation is needed. ``seq_model`` selects the sequence
    model over the frequency tokens ('lstm' | 'mamba' | 'bimamba'), reusing the same
    blocks as ``FreqGroupLSTM`` -- the Mamba imports stay lazy so the LSTM/CPU path is
    unaffected.
    """
    def __init__(self, c_in=1, embedding=128, patch_t=1, patch_f=BINS_PER_SEMITONE,
                 n_freq_out=88, seq_model='lstm', mamba_impl='mamba1', depth=2) -> None:
        super().__init__()

        # Patch embedding: non-overlapping (patch_t x patch_f) patches -> embedding dim.
        self.patch_embed = nn.Conv2d(c_in, embedding,
                                     kernel_size=[patch_t, patch_f],
                                     stride=[patch_t, patch_f])
        # Learnable positional embedding over the 88 frequency tokens (fixed pitch semantics).
        self.pos_embed = nn.Parameter(torch.zeros(1, embedding, 1, n_freq_out))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.InstanceNorm2d(embedding)

        # Sequence model over the frequency axis, stacked `depth` times.
        # Mirrors FreqGroupLSTM's selection so in == out == embedding for each block.
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if seq_model == 'lstm':
                self.blocks.append(BiLSTM(embedding, embedding//2))
            elif seq_model == 'mamba':
                from .mamba import MambaSeq
                self.blocks.append(MambaSeq(embedding, embedding, impl=mamba_impl))
            elif seq_model == 'bimamba':
                from .mamba import BiMambaSeq
                self.blocks.append(BiMambaSeq(embedding, embedding, impl=mamba_impl))
            else:
                raise ValueError(f'unknown seq_model: {seq_model}')

    def forward(self, x):
        # inputs:  [b x 1 x T x 352]
        # outputs: [b x embedding x T x 88]

        # => [b x C x T x 88]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.norm(x)

        b, c, t, n_freq = x.size()
        # => [b x T x freq x C]
        x = torch.permute(x, [0, 2, 3, 1])
        # => [(b*T) x freq x C] -- each time frame is an independent sequence over frequency
        x = x.reshape([b*t, n_freq, c])
        for blk in self.blocks:
            # => [(b*T) x freq x C]
            x = blk(x)
        # => [b x T x freq x C]
        x = x.reshape([b, t, n_freq, c])
        # => [b x C x T x 88]
        x = torch.permute(x, [0, 3, 1, 2])
        return x


class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 152])
    def forward(self, x):
        x = self.conv_1(x) + self.conv_2(x) + self.conv_3(x) + self.conv_4(x) +\
            self.conv_5(x) + self.conv_6(x) + self.conv_7(x) + self.conv_8(x)
        x = torch.relu(x)
        return x


class CNNTrunk(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out)
            )

    def __init__(self, c_in = 1, c_har = 16,  embedding = 128, fixed_dilation = 24) -> None:
        super().__init__()

        self.block_1 = self.get_conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = self.get_conv2d_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(c_har, c_har, kernel_size=7)

        c3_out = embedding
        
        self.conv_3 = HarmonicDilatedConv(c_har, c3_out)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        # self.conv_9 = nn.Conv2d(c3_out, 64,1)
        # self.conv_10 = nn.Conv2d(64, 1, 1)

    def forward(self, log_gram_db):
        # inputs: [b x 2 x T x n_freq] , [b x 1 x T x 88]
        # outputs: [b x T x 88]


        # img_path = 'logspecgram_preview.png'
        # if not os.path.exists(img_path):
        #     img = torch.permute(log_gram_db, [2, 0, 1]).reshape([352, 640*4]).detach().cpu().numpy()
        #     # x_grid = torchvision.utils.make_grid(x.swapaxes(0, 1), pad_value=1.0).swapaxes(0, 2).detach().cpu().numpy()
        #     # plt.imsave(img_path, (x_grid+80)/100)
        #     plt.imsave(img_path, img)

        # => [b x 1 x T x 352]
        # x = torch.unsqueeze(log_gram_db, dim=1)



        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)

        return x