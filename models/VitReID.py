import os
import torch
import torch.nn as nn
from .backbone.vit_pytorch import vit_base_patch16_224_TransReID

def weights_init_kaiming(m):
    
    classname = m.__class__.__name__
    
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class TransReIDBase_Inference(nn.Module):
    
    def __init__(self, cfg) -> None:

        super(TransReIDBase_Inference, self).__init__()

        self.neck_feature = cfg.TEST.NECK_FEAT

        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TEST, 
            aie_xishu=cfg.MODEL.AIE_COE,
            local_feature=cfg.MODEL.LOCAL_F,
            stride_size=cfg.MODEL.STRIDE_SIZE, 
            drop_path_rate=cfg.MODEL.DROP_PATH
        )
        self.in_planes = self.base.embed_dim
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.load_param(cfg.TEST.WEIGHT)      

    def load_param(self, trained_path:os.PathLike):
        print(f'Loading pretrained reid_model from {trained_path}')
        param_dict = torch.load(trained_path,map_location='cpu')
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'gap' in i:
                continue
            k = i if 'module' not in i else i.replace('module.', '')
            self.state_dict()[k].copy_(param_dict[i])
            
    @torch.no_grad()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        global_feat = self.base(x)
        if self.neck_feature == "after":
            return self.bottleneck(global_feat) 
        return global_feat
        