#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Config, T5Model, T5ForConditionalGeneration
from models.audio_encoder_config import AudioEncoderConfig
from audio_encoder import AudioEncoderModel


class T5CaptionModel(nn.Module):

    def __init__(self, config):
        super(T5CaptionModel, self).__init__()
        #encoder config1
        self.config = config
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)
           
        #decoder
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.decoder_start_token_id = self.decoder.decoder.config.decoder_start_token_id


    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward_encoder(self, audio):
        outputs = self.encoder(audio) #(1,token,1024)
        return outputs

    def forward_decoder(self, text, encoder_outputs):
        # encoder_outputs = self.encoder(encoder_outputs)["last_hidden_state"]
        
        #generate text
        text = self.tokenizer(text,
                              padding='max_length',
                              truncation=True,
                              max_length=256,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device) #torch.Size([batch, token_len])
        attention_mask = text["attention_mask"].to(self.device) #torch.Size([batch, token_len])

        decoder_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        decoder_input_ids = self.shift_tokens_right(decoder_targets, self.decoder.config.pad_token_id, self.decoder_start_token_id
        ) #（teacher forcing）

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=None,
            encoder_outputs=(encoder_outputs,),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
        )
        lm_logits = decoder_outputs["logits"]
        
        loss = self.loss_fct(lm_logits.view(-1, 32128), decoder_targets.view(-1)) 
        return loss
    
    def forward(self, audio, text):

        audio_embeds = self.forward_encoder(audio)
        loss = self.forward_decoder(text, audio_embeds)   
        return loss 

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=1,
                 max_length=256,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        encoder_outputs = self.encoder(samples)

        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.decoder_start_token_id # batch x 1
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device) # batch x 1

        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                head_mask=None,
                decoder_head_mask=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty) 

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [cap.replace(". ", ".") for cap in captions]
        return captions

if __name__ == '__main__':
    import yaml
    with open("settings/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = T5CaptionModel(config)
    print(model)
    audio_feats = torch.randn(1, 160000)
    text = ["this is a sample"]
    # loss = model.forward_decoder(text, audio_feats)
    output = model.generate(audio_feats)
    print(output)