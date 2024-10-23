#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import numpy as np
import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer
from transformers import BertTokenizer, AutoConfig, BertConfig, BertLMHeadModel
from models.audio_encoder_config import AudioEncoderConfig
from audio_encoder import AudioEncoderModel
from models.configuration_audio_encoder_decoder import AudioEncoderDecoderConfig
from models.modeling_audio_encoder_decoder import AudioEncoderDecoderModel


class BertCaptionModel(nn.Module):

    def __init__(self, config):
        super(BertCaptionModel, self).__init__()
        #encoder config1
        self.config = config
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],audio_args=config["audio_args"])

        #decoder config1
        decoder_config = AutoConfig.from_pretrained("bert-base-uncased", add_cross_attention=True, is_decoder=True)

        # two config mergue2
        self.model_config = AudioEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config,decoder_config)
                               
        #model
        self.model = AudioEncoderDecoderModel(config=self.model_config, is_pretrained=False)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_config.decoder_start_token_id = self.tokenizer.cls_token_id


    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_encoder(self, audios):
        # samples: audio features
        audio_embed = self.model.encoder(audios)['last_hidden_state']
        return audio_embed

    def forward_decoder(self, text):
        # text = [s.strip().replace('heard between ','').replace(' and','') for s in text]
        text = self.tokenizer(text,
                        padding="longest",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        # attention_mask = text["attention_mask"].to(self.device)
        # decoder_targets = input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        # decoder_targets[:, 0] = -100
        text_embed = self.model.decoder.bert(input_ids)["last_hidden_state"]

        return text_embed
    
    def compute_similarity(self, audio_embed, query_embed):
        """
        Compute an audio-text alignment matrix.
        :param audio_embed: tensor, (T, E).
        :param query_embed: tensor, (Q, E).
        :return: similarity matrix: tensor, (T, Q).
        """

        # Compute dot products
        sim = torch.mm(audio_embed, query_embed.t())  # [T, Q]
        sim = torch.clamp(sim, min=0.)
        
        return sim.mean()

    def forward(self, audio, text):
        text_token = self.tokenizer(text,
                              padding="longest",
                              truncation=True,
                              max_length=256,
                              return_tensors="pt")
        #generate bertLMhead
        input_ids = text_token["input_ids"].to(self.device)
        attention_mask = text_token["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)

        decoder_targets[:, 0] = -100

        decoder_output = self.model(
            audio_feats=audio,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            labels=decoder_targets,
            return_dict=True
        )

        return decoder_output.loss 

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=1,
                 max_length=256,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):
        # samples: audios

        if use_nucleus_sampling:
            outputs = self.model.generate(
                inputs=samples,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                decoder_start_token_id=self.model_config.decoder_start_token_id
                    )
        else:
            outputs = self.model.generate(
                inputs=samples,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                decoder_start_token_id=self.model_config.decoder_start_token_id
            )

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [cap.replace(". ", ".") for cap in captions]
        return captions


if __name__ == '__main__':
    with open("settings/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = BertCaptionModel(config)
    print(model)
    audio_feats = torch.randn(2, 160000)
    text = ["cat heard between 0.0 and 10.0 seconds","cat heard between 0.0 and 10.0 seconds"]
    loss = model(audio_feats, text)
    print(loss)

    # output = model.generate(audio_feats)
    # print(output)