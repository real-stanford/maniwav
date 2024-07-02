import torch.nn as nn
import torchaudio
import numpy as np
import random
import torch
import os

import torchaudio.functional as F
from audiomentations import *

class RobotNoiseAug(nn.Module):
    def __init__(self, noise_path, key=None, p=1.0):
        super().__init__()
        self.noise = np.array(torchaudio.load(noise_path)[0])
        self.key = key
        self.p = p
        
    def forward(self, in_audio):
        waveform = np.reshape(in_audio, -1)
        audio_length = len(waveform)
        noise_length = self.noise.shape[-1]
        
        idx = np.random.choice(np.arange(0, noise_length-audio_length))
        assert self.key is not None
        noise_clip = self.noise[0, idx:idx+audio_length]
        
        if random.random() < self.p:
            output = waveform + noise_clip
        else:
            output = waveform
        return output


class NoiseAug(nn.Module):
    def __init__(self, noise_folder_path, key=None, p=1.0):
        super().__init__()
        self.noises = []
        for noise_name in os.listdir(noise_folder_path):
            self.noises.append(np.array(torchaudio.load(os.path.join(noise_folder_path, noise_name))[0]))
        self.key = key
        self.p = p
        
    def forward(self, in_audio):
        waveform = np.reshape(in_audio, -1)
        audio_length = len(waveform)
        
        noise_idx = np.random.choice(np.arange(0, len(self.noises)))
        noise_length = self.noises[noise_idx].shape[-1]
        
        idx = np.random.choice(np.arange(0, noise_length-audio_length))
        noise_clip = self.noises[noise_idx][0, idx:idx+audio_length]
        
        if random.random() < self.p:
            output = waveform + noise_clip
        else:
            output = waveform
        return output


if __name__ == '__main__':
    robot_noise = torchaudio.load('data/robot-noise-calib/robot.wav')[0][0, 48000*5:48000*8]
    audio = torchaudio.load('/home/liuzeyi/Desktop/universal_manipulation_interface/data/20240529_flipping_v1/demos/demo_C3464250587589_2022.01.05_23.14.56.129183/audio.wav')[0][0, :48000*3]
    
    noise_folder_path = 'data/esc-50'
    for i, noise_name in enumerate(os.listdir(noise_folder_path)):
        noise = torchaudio.load(os.path.join(noise_folder_path, noise_name))[0][0, :len(audio)]
        ratio = (torch.max(audio) - torch.min(audio)) / ((torch.max(noise) - torch.min(noise)))
        output = audio + noise + robot_noise
        output = output.reshape(1, len(audio))
        torchaudio.save(f'{noise_name}_aug.wav', output, 48000)
        # break