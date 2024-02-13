# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    require diffusers-0.11.1
"""
import os
import clip
import time
import torch
from PIL import Image
from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points
from models.vae_adain import Model as VAE

# model_path = './lion_ckpt/unconditional/airplane/checkpoints/model.pt'
model_path = './lion_ckpt/unconditional/all55/checkpoints/epoch_10999_iters_2100999.pt'
# model_path = './lion_ckpt/unconditional/all55/checkpoints/vae_only.pt'
# model_config = './lion_ckpt/unconditional/airplane/cfg.yml'
model_config = './lion_ckpt/unconditional/all55/cfg.yml'

num_sample = 10

config.merge_from_file(model_config)
if 'vae_only' in model_path:
    lion = VAE(config).cuda()
    ckpt = torch.load(model_path)
    lion.load_state_dict(ckpt['model'])
else:
    lion = LION(config)
    lion.load_model(model_path)

if config.clipforge.enable:
    input_t = ["a swivel chair, five wheels"] 
    device_str = 'cuda'
    clip_model, clip_preprocess = clip.load(
                        config.clipforge.clip_model, device=device_str)    
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    print('clip_feat', clip_feat.shape)
else:
    clip_feat = None

start_time = time.time()
if 'vae_only' in model_path:
    # decomposed_eps = [torch.rand(num_sample, 128, 1, 1), torch.rand(num_sample, 8192, 1, 1)] # [0]: global prior, [1]: local prior
    output = lion.sample(num_sample)
    pts = output
else:
    output = lion.sample(num_sample if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
    pts = output['points']
end_time = time.time()

print("Time elapsed:", str(end_time - start_time))

img_name = "demo.png"
plot_points(pts, output_name=img_name)
img = Image.open(img_name)