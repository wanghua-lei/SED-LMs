#!/usr/bin/env python3.9


import json
import random
import librosa
import torch
import ruamel.yaml as yaml
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from data_handling.datamodule import collate_fn
from data_handling.sampler import BySequenceLengthSampler, BySequenceBatchSampler
from data_handling.text_transform import text_preprocess


def _load_json_file(files):
    json_data = []
    audio_id = 0
    for file in files:
        with open(file, "r") as f:
            json_obj = json.load(f)
            for item in json_obj["data"]:
                temp_dict = {"audio": item["audio"], "caption": item[f"caption"]}# "id": audio_id
                json_data.append(temp_dict)
            audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, json_files, audio_config):

        self.json_data = _load_json_file(json_files)
        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        item = self.json_data[index]
        wav_path = item["audio"]
        try:
            waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)
            if self.max_length != 0:
                # if audio length is longer than max_length, we randomly crop it to mac length
                if waveform.shape[-1] > self.max_length:
                    max_start = waveform.shape[-1] - self.max_length
                    start = random.randint(0, max_start)
                    waveform = waveform[start: start + self.max_length]
        except:
            print("Error loading audio: ", wav_path)
            return self.__getitem__(index+1)

        caption = text_preprocess(item["caption"])
        audio_id = index
        return torch.tensor(waveform), caption, "", audio_id


def pretrain_dataloader(config,
                        # bucket: bool = True,
                        is_distributed: bool = False,
                        num_tasks: int = 0,
                        global_rank: int = 0):
    dataset = AudioLanguagePretrainDataset(config["json_files"], config["audio_args"])
    if is_distributed:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     shuffle=True)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=config["data_args"]["batch_size"],
        num_workers=config["data_args"]["num_workers"],
        pin_memory=True,
        sampler=sampler,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


if __name__ == '__main__':

    with open("../settings/pretrain.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
