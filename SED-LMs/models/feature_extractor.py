#!/usr/bin/env python3.9


import torch
import torch.nn as nn
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
import os
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tools.rrc_module import RandomResizeCrop
import random
class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(CustomAudioTransform):
    def __init__(self, min, max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) *2. - 1.
        return input
    

class ATSTNorm(nn.Module):
    def __init__(self):
        super(ATSTNorm, self).__init__()
        # Audio feature extraction
        self.amp_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.scaler = MinMax(min=-79.6482,max=50.6842) # TorchScaler("instance", "minmax", [0, 1])

    def amp2db(self, spec):
        return self.amp_to_db(spec).clamp(min=-50, max=80)

    def forward(self, spec):
        spec = self.scaler(self.amp2db(spec))
        return spec
    

class AudioFeature(nn.Module):
    def __init__(self, audio_config):
        super().__init__()
        self.mel_trans = Spectrogram(n_fft=audio_config["n_fft"], 
                                     hop_length=audio_config["hop_length"],
                                     win_length=audio_config["n_fft"],
                                     window='hamming',
                                     center=True,
                                     pad_mode='reflect',
                                     freeze_parameters=True)

        self.log_trans = LogmelFilterBank(sr=audio_config["sr"],
                                          n_fft=audio_config["n_fft"],
                                          n_mels=audio_config["n_mels"],
                                          fmin=audio_config["f_min"],
                                          fmax=audio_config["f_max"],
                                          ref=1.0,
                                          amin=1e-10,
                                          top_db=None,
                                          freeze_parameters=True)
        self.to_db = ATSTNorm()
        self.freq_warp = RandomResizeCrop((1,1.0),time_scale=(1.0,1.0))

    def forward(self, input):
        mel_feats = self.mel_trans(input)
        log_mel = self.log_trans(mel_feats)
        mel_norm = self.to_db(log_mel)

        return mel_norm


class ATSTTransform:
    def __init__(self):
        self.transform = MelSpectrogram(16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64, window_fn=torch.hann_window,power=1)
        self.to_db = ATSTNorm()
        self.freq_warp = RandomResizeCrop((1,1.0),time_scale=(1.0,1.0))

    def __call__(self, x):
        # to_db applied in the trainer files
        x = self.transform(x)
        return self.to_db(x.cpu())
    
    
    
