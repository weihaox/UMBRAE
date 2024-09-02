#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2024/01/14 18:25:30
@Author  :   Weihao Xia 
@Version :   2.0
@Desc    :   None
'''

from functools import partial

from transformers import CLIPVisionModel 
from perceiver import PerceiverResampler

import torch
from torch import nn
from torchvision import transforms


class BrainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')        
        self.clip_size = (224, 224)       
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            # transforms.ToTensor(), # only for debug
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc

        for param in self.clip.parameters():
            param.requires_grad = False
            # param.data = param.data.half()

        self.clip_width = self.clip.vision_model.embeddings.patch_embedding.out_channels

        self.conv1 = nn.ModuleDict()
        self.position_embedding = nn.ParameterDict()
        self.modals = ['image', 'fmri']
        for modal in self.modals:
            if modal =='image':
                modal_tokens = 256 + 1
                pass
            elif modal == 'fmri':
                modal_tokens = 8 + 1
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.position_embedding[modal] = nn.Embedding(modal_tokens, self.clip_width)

    def clip_encode_image(self, x, modal='image'):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 

        x = torch.cat([self.clip.vision_model.embeddings.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  

        pos_embedding = self.clip.vision_model.embeddings.position_embedding # Embedding(257, 1024)
        if modal == 'fmri':
            pos_embedding = self.position_embedding[modal]
            
        modal_tokens = 257 if modal == 'image' else 9
        position_ids =  torch.arange(0, modal_tokens).unsqueeze(0).to(x.device)

        x = x + pos_embedding(position_ids)
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)

        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer] # torch.Size([1, 257, 1024])
        image_features = select_hidden_state[:, 1:] # torch.Size([1, 256, 1024]

        return image_features

    def encode_image(self, x, modal='image'):
        if modal in ['image']:
            x = self.preprocess(x)
            x = self.clip.vision_model.embeddings.patch_embedding(x)  # conv1, shape = [*, width, grid, grid]
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            x = x.reshape(x.size(0), self.clip_width, -1)

        image_feats = self.clip_encode_image(x, modal=modal)

        return image_feats


class Perceiver(nn.Module):
    def __init__(self, patch_embed_dim=1024, hidden_size=4096, num_latents=256):
        super().__init__()
    
        self.ln_vision = nn.LayerNorm(patch_embed_dim)
        self.llm_proj = nn.Linear(
            patch_embed_dim, hidden_size
        )

        self.perceiver = PerceiverResampler(
            dim = patch_embed_dim,
            dim_head = 96,
            depth = 6,
            heads = 16,
            num_latents = num_latents,
            num_media_embeds = 1
        )

    def forward(self, image_features):
        image_features = self.ln_vision(image_features)
        inputs_llm = self.perceiver(image_features)
        return self.llm_proj(inputs_llm)

# brain encoder for corss-subject training and inference
class BrainX(nn.Module):
    def __init__(self, hidden_dim=1024, out_dim=1024, num_latents=256, use_token=False, use_norm=False, act_first=False):
        super().__init__()
        self.subs = [1, 2, 5, 7]
        self.num_voxels = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
        self.use_token = use_token

        norm_func = partial(nn.LayerNorm, normalized_shape=num_latents)
        act_fn = nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.lin1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        self.token = nn.ParameterDict()
        for sub in self.subs:
            if not use_norm:
                self.lin1[f'fmri{sub}']  = nn.Linear(1, num_latents)
            else:
                self.lin1[f'fmri{sub}'] = nn.Sequential(
                    nn.Linear(1, num_latents),
                    *[item() for item in act_and_norm],
                    nn.Dropout(0.5),
                    )
            self.lin2[f'fmri{sub}']  = nn.Linear(self.num_voxels.get(sub), hidden_dim)

            if self.use_token:            
                self.token[f'fmri{sub}'] = nn.Parameter(
                    torch.empty([1, 5, hidden_dim])) # learnable tokens dim=5
                nn.init.normal_(self.token[f'fmri{sub}'], std=0.02) 

        self.perceiver = Perceiver(patch_embed_dim=hidden_dim, hidden_size=out_dim, num_latents=num_latents)          

    def forward(self, x, modal='fmri1'):
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.lin1[modal](x)
        
        x = x.transpose(1, 2)
        x = self.lin2[modal](x)
        if self.use_token:
            token = self.token[modal].repeat(x.size(0), 1, 1)
            x = torch.cat([x, token], dim=1)
        x = self.perceiver(x)
        return x
    
# brain encoder for single-subject training and inference
class BrainXS(nn.Module):
    def __init__(self, in_dim=15724, hidden_dim=1024, out_dim=1024, num_latents=256):
        super().__init__()
        self.lin1 = nn.Linear(1, num_latents)
        self.lin2 = nn.Linear(in_dim, hidden_dim)

        self.perceiver = Perceiver(patch_embed_dim=hidden_dim, hidden_size=out_dim, num_latents=num_latents)
        
    def forward(self, x):
        x = x.unsqueeze(1) # [B, 1, 15724]
        x = x.transpose(1, 2)
        x = self.lin1(x)
        
        x = x.transpose(1, 2)
        x = self.lin2(x)
        x = self.perceiver(x)
        return x

# brain encoder for weakly-supervised adaptation (data-efficient training)
class BrainXC(nn.Module):
    def __init__(self, hidden_dim=1024, out_dim=1024, num_latents=256, sub=7, freeze_encoder=True):
        super().__init__()
        self.num_voxels = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}

        self.lin1 = nn.Linear(1, num_latents)
        self.lin2 = nn.Linear(self.num_voxels.get(sub), hidden_dim)
        self.perceiver = Perceiver(patch_embed_dim=hidden_dim, hidden_size=out_dim, num_latents=num_latents)  

        if freeze_encoder:
            for param in self.perceiver.parameters():
                param.requires_grad = False
                # param.data = param.data.half()    

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.lin1(x)
        
        x = x.transpose(1, 2)
        x = self.lin2(x)
        # with torch.no_grad():
        x = self.perceiver(x)
        return x