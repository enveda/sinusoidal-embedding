import numpy as np

import torch
from torch import nn

class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0, activation=None):
        super(FeedForwardBlock, self).__init__()

        hidden_dim = out_dim if hidden_dim is None else hidden_dim

        self._linear_1 = nn.Linear(in_dim, hidden_dim)
        self._linear_2 = nn.Linear(hidden_dim, out_dim)

        self._activation = nn.ReLU() if activation is None else activation

        self._dropout_1 = nn.Dropout(dropout)
        self._dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self._dropout_1(self._activation(self._linear_1(x)))
        x = self._dropout_2(self._linear_2(x))

        return x

class MultiLayerFeedForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0, dropout_last_layer=True):
        super(MultiLayerFeedForwardBlock, self).__init__()

        self._layers = list()

        if not hasattr(hidden_dim,  '__iter__'):
            if hidden_dim is None:
                hidden_dim = [out_dim]
            else:
                hidden_dim = [hidden_dim]

        layer_dims = [in_dim] + hidden_dim + [out_dim]

        for i in range(1, len(layer_dims) - 1):
            self._layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            self._layers.append(nn.ReLU())
            self._layers.append(nn.Dropout(dropout))

        self._layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        if dropout_last_layer:
            self._layers.append(nn.Dropout(dropout))

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x)

class SinusoidalMZEmbedding(nn.Module):
    def __init__(
            self, embd_dim, mz_log_lims=None, mz_spacing=None, mz_precision=None, dropout=None,
            hidden_dim=None):
        super(SinusoidalMZEmbedding, self).__init__()

        mz_log_lims = (-2, 3) if mz_log_lims is None else mz_log_lims
        mz_spacing = 'log' if mz_spacing is None else mz_spacing
        mz_precision = 32 if mz_precision is None else mz_precision
        dropout = 0.0 if dropout is None else dropout

        self._embd_dim = embd_dim

        if mz_spacing == 'log':
            frequency = 2 * np.pi / torch.logspace(
                mz_log_lims[0], mz_log_lims[1], int(embd_dim / 2), dtype=torch.float64)
        elif mz_spacing == 'linear':
            frequency = 2 * np.pi / torch.linspace(
                mz_log_lims[0], mz_log_lims[1], int(embd_dim / 2), dtype=torch.float64)
        else:
            raise ValueError('mz_spacing must be either log or linear')

        self._frequency = nn.Parameter(frequency, requires_grad=False)

        self._ff_block = MultiLayerFeedForwardBlock(
            embd_dim, embd_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, mz_tensor):
        omega_mz = self._frequency.reshape((1, 1, -1)) * mz_tensor
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
        mz_vecs = torch.cat([sin, cos], axis=2)

        return self._ff_block(mz_vecs.to(torch.float32))

class PeakEmbedding(nn.Module):
    def __init__(self, mz_embd, dropout=0.0, hidden_dim=None, drop_precursor=False):
        super(PeakEmbedding, self).__init__()

        self._mz_embd = mz_embd
        self._embd_dim = self._mz_embd._embd_dim
        self._ff_block = FeedForwardBlock(
            self._embd_dim + 1, self._embd_dim, hidden_dim=hidden_dim, dropout=dropout)
        self._drop_precursor = drop_precursor

    def forward(self, data):
        mz_tensor, intensity_tensor = data

        if self._drop_precursor:
            mz_tensor = mz_tensor[:, 1:]
            intensity_tensor = intensity_tensor[:, 1:, :]

        embd = self._mz_embd(mz_tensor)
        embd = torch.cat([embd, intensity_tensor], axis=2)
        embd = self._ff_block(embd)

        return embd
