
import torch
import yaml
from transformers.modeling_outputs import BaseModelOutput
from models.cnns import Cnn10, Cnn14, ResNet38
from models.htsat import HTSAT_Swin_Transformer
from models.atst.atst_model import ATST
from transformers import PreTrainedModel
from models.audio_encoder_config import AudioEncoderConfig
from models.beats.beats_model import BEATs_model

class AudioEncoderModel(PreTrainedModel):
    config_class = AudioEncoderConfig

    def __init__(self, config):
        super(AudioEncoderModel, self).__init__(config)

        if config.model_arch == "cnn":
            if config.model_name == 'ResNet38':
                self.audio_enc = ResNet38(config)
            elif config.model_name == 'Cnn14':
                self.audio_enc = Cnn14(config)
            elif config.model_name == 'Cnn10':
                self.audio_enc = Cnn10(config)

            if config.pretrained:
                # loading pretrained CNN weights
                pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                            format(config.model_name))['model']
                dict_new = self.audio_enc.state_dict().copy()
                trained_list = [i for i in pretrained_cnn.keys()
                                if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
                for i in range(len(trained_list)):
                    dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
                self.audio_enc.load_state_dict(dict_new)
                # print("Weights loaded for audio encoder.")
            self.audio_width = 1024
        elif config.model_arch == "transformer":
            if config.model_name == 'htsat':
                self.audio_enc = HTSAT_Swin_Transformer(
                    spec_size=256,
                    patch_size=4,
                    patch_stride=(4, 4),
                    num_classes=527,
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[4, 8, 16, 32],
                    window_size=8,
                    config=config
                )
                if config.pretrained:
                    audio_ckpt = torch.load("pretrained_models/audio_encoder/HTSAT.ckpt", map_location="cpu")["state_dict"]
                    for key in list(audio_ckpt.keys()):
                        if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                            and 'logmel_extractor' not in key):
                            v = audio_ckpt.pop(key)
                            audio_ckpt[key[10:]] = v
                    self.audio_enc.load_state_dict(audio_ckpt, strict=False)
                    # param_names = [n for n, p in self.audio_enc.named_parameters()]
                    # for n in param_names:
                    #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
                self.audio_width = 768
            elif config.model_name == 'atst':
                if config.pretrained:
                    path = 'SED-LMs/ATST-SED/ckpts/atst_as2M.ckpt'
                    self.audio_enc = ATST(path, atst_dropout=0.0, config=config)
                self.audio_width = 768
            elif config.model_name == 'beats':
                if config.pretrained:
                    path = "SED-LMs/pretrained_models/BEATs_iter3_plus_AS2M.pt"
                    self.audio_enc = BEATs_model(path)
                self.audio_width = 768
  

        else:
            raise NotImplementedError('No such audio encoder network.')


        if config.freeze:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):
        if self.config.model_name == 'beats':
            audio_embeds = self.audio_enc(input_ids,input_ids)
        else:
            audio_embeds = self.audio_enc(input_ids)
        # attention_weights = [attn for _, attn in att_map]
        if not return_dict:
            return (audio_embeds, )
        # if output_attentions==True:
        #     return attention_weights
        return BaseModelOutput(audio_embeds, None, None)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    with open("settings/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["audio_encoder_args"]["model_name"] = 'htsat'

    config_encoder = AudioEncoderConfig(**config["audio_encoder_args"], audio_args=config["audio_args"])
    model = AudioEncoderModel(config_encoder)
    x = torch.randn(1, 160000)

    output = model(x)
    print(output.last_hidden_state.shape)

