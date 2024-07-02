# Copied and modified from https://github.com/JunzheJosephZhu/see_hear_feel/blob/master/src/models/encoders.py
import torch
import torchaudio
import torch.nn as nn
import timm
from matplotlib import pyplot as plt
import librosa
import pickle

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from diffusion_policy.common.pytorch_util import replace_submodules


class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""

    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class AudioEncoder(nn.Module):
    def __init__(self, model, norm_spec, audio_key='mic_0'):
        super().__init__()
        self.model = model
        self.coord_conv = CoordConv() # similar as positional encoding
        self.norm_spec = norm_spec
        self.stats = None
        self.audio_key = audio_key
    
    def forward(self, spec):
        EPS = 1e-8
        # spec: B x C x Mel x T
        log_spec = torch.log(spec + EPS)
        assert log_spec.size(-2) == 64
        x = log_spec
        if self.norm_spec.is_norm:
            x = (x - self.norm_spec.min) / (self.norm_spec.max - self.norm_spec.min)
            x = x * 2 - 1

        x = self.coord_conv(x)
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = timm.create_model(
            model_name='resnet18',
            pretrained=True,
            global_pool='avg', # '' means no poling
            num_classes=0            # remove classification layer
        )
    model = replace_submodules(
        root_module=model,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                num_channels=x.num_features)
        )
    audio_encoder = AudioEncoder(model)
    signal, sr = torchaudio.load("data/audio.wav")
    signal = signal[:, :96000]
    print("input shape:", signal.unsqueeze(0).shape)
    output = audio_encoder(signal.unsqueeze(0))
    print("output shape:", output.size())