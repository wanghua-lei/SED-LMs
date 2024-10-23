#!/usr/bin/env python3.9
import os
import json
import random
import warnings
import librosa
import numpy
import torch
import re
from torch.utils.data import Dataset
from data_handling.text_transform import text_preprocess, text_enhance
import random
warnings.filterwarnings("ignore")

class AudioCaptionDataset(Dataset):

    def __init__(self,
                 audio_config: dict,
                 dataset: str = "DCASE",
                 split: str = "train",
                 ):
        super(AudioCaptionDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.sr = audio_config["sr"]
        json_path = f"data/{dataset}/json_files/{split}.json"

        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0
        with open(json_path, 'r') as f:
            json_obj = json.load(f)["data"]
        if split in ["train","test","val"]: 
            self.captions = [item["caption"] for item in json_obj]
            self.wav_paths = [item["audio"] for item in json_obj]
        else:
            self.wav_paths = [item["audio"] for item in json_obj]


    def __len__(self):
        return len(self.wav_paths)


    def __getitem__(self, index):
        audio_idx = index
        audio_name = self.wav_paths[index].split("/")[-1]
        wav_path = self.wav_paths[index]
        try:
            waveform, sr = librosa.load(wav_path, sr=self.sr, mono=True)
            if self.max_length != 0:
                # if audio length is longer than max_length, we random crop it
                if waveform.shape[-1] > self.max_length:
                    max_start = waveform.shape[-1] - self.max_length
                    start = random.randint(0, max_start)
                    waveform = waveform[start: start + self.max_length]
                else:
                    # if audio length is shorter than max_length, we pad it
                    waveform = numpy.pad(waveform, (0, self.max_length - waveform.shape[-1]), "constant")
        except:
            print("Error loading audio: ", wav_path)
            return self.__getitem__(index+1)

        #load caption
        if self.split in ["train","test","val"]:
            if random.random() < 0.5:
                caption = text_preprocess(text_enhance(self.captions[index]))
            else:
                caption = text_preprocess(self.captions[index])
            return torch.from_numpy(waveform), caption, audio_name, audio_idx
        
        

