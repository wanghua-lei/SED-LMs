import os
import torch
from .audio_transformer import FrameASTModel
from models.feature_extractor import ATSTTransform



class ATST(torch.nn.Module):
    def __init__(self, atst_path, *args, atst_dropout=0.0, config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.atst = FrameASTModel(atst_dropout=atst_dropout)
        self.load_atst(atst_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None
        self.audio_feats_extractor = ATSTTransform() #config.audio_args

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, atst_feat, other_emb=None):
        
        atst_mel = self.audio_feats_extractor(atst_feat)# [1, 1001, 64]
        #atst_feat = atst_mel.transpose(1, 2)
        atst_feat = atst_mel.unsqueeze(1)
        atst_x = self.atst.get_intermediate_layers(
            atst_feat,
            self.fake_length.to(atst_feat).repeat(len(atst_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        #atst_x = atst_x.transpose(1, 2)
        return atst_x #[1,512,768]


    def load_atst(self, path=None):
        if path is None:
            pre_path = "./ckpts/atst_as2M.ckpt" 
            assert os.path.exists(pre_path), "Please make sure you have a default path to load ATST. Please change this path to the atst_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the atst_as2M.ckpt the downloaded checkpoint from the home page.
        
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        atst_state_dict = {}
        for k, v in state_dict.items():
            #print(k)
            # if "model.teacher.encoder." in k:
            #     if "encoder.norm." in k:
            #         new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
            #     elif "cls_token" in k:
            #         continue
            #     else:
            #         new_k = k.replace("model.teacher.encoder.", "")
            #     atst_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                atst_state_dict[new_k] = v
                continue
            # if "encoder.encoder.teacher_module." in k:
            #     continue
            # ATST-Frame
            if "encoder.encoder." in k:
                new_k = k.replace("encoder.encoder.", "")
                atst_state_dict[new_k] = v

        self.atst.load_state_dict(atst_state_dict, strict=False)
        # for n, param in self.atst.named_parameters():
        #     param.requires_grad = True
        # state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
        # atst_state_dict = {}
        # for k, v in state_dict.items():
        #     if ("total_ops" in k) or ("total_params" in k):
        #         continue
        #     if "atst_frame.atst." in k:
        #         k = k.replace("atst_frame.atst.", "")
        #         atst_state_dict[k] = v
        # self.atst.load_state_dict(atst_state_dict, strict=True)
        # for n, param in self.atst.named_parameters():
        #     param.requires_grad = False
