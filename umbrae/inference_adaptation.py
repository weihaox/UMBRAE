#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   inference_adaptation.py
@Time    :   2024/02/25 11:58:30
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   inference script for adaptation
@usage   :   
for ratio in 0.05 0.5 1.0
do
python inference_adaptation.py \
    --prompt 'Describe this image <image> as simply as possible.' \
    --data_path '/home/wx258/project/nsd_data' --fmri_encoder 'brainxc' --subj 7 \
    --tokenizer_path 'train_logs/demo_weak_adaptation/brainx_adaptation_7_${ratio}/last.pth' \
    --encoder_path 'train_logs/demo_cross_subject/brainx_adaptation_125/last.pth' \
    --save_path 'braingrounding/evaluation/eval_caption/caption_results/prompt_4/result_240222_unibrain_continual/sub07_dim1024_${ratio}'
done
'''

import os
import json
import time
import argparse
import braceexpand
import webdataset as wds
import utils
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torchvision.transforms import ToPILImage
from model import BrainXC
from utils import postprocess

parser = argparse.ArgumentParser()
parser.add_argument('--shikra_path', default='model_weights/shikra-7b')
parser.add_argument('--adapter_path', default='model_weights/mm_projector.bin')
parser.add_argument('--tokenizer_path', type=str, help='path to the unibrain tokenizer', required=True)
parser.add_argument('--encoder_path', type=str, help='path to the unibrain perceive encoder', required=True)
parser.add_argument('--fmri_encoder', type=str, default='brainxc', help='type of brainnet', choices=['brainxc'])
parser.add_argument('--use_norm', type=bool, default=False, help='whether to use norm layer in the model')
parser.add_argument('--use_token', type=bool, default=False, help='whether to use learnable token in the model')
parser.add_argument('--feat_dim', type=int, help='output dimension of the fmri encoder', default=1024, choices=[1024, 4096])
parser.add_argument('--data_path', type=str, default="nsd_data", help='path to nsd data')
parser.add_argument('--save_path', type=str, default='results', help='path to save results')
parser.add_argument('--save_image', type=bool, default=False, help='save image or not')
parser.add_argument('--prompt', required=True, help='prompt for the model')
parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed, cudnn_deterministic=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# prepare models and data loaders
print('prepare NSD webdataset data...')
val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_val = 982

# result_dir = os.path.join(os.path.dirname(__file__), 'results/sub{:02d}_dim{}'.format(subj, feat_dim)) 
result_dir = os.path.join(save_path, 'sub{:02d}_dim{}'.format(subj, feat_dim)) 
os.makedirs(result_dir, exist_ok=True)

# save config in a json file
args_dict = vars(args)
with open(os.path.join(result_dir, 'config.json'), 'w') as file:
    json.dump(args_dict, file, indent=4)

print('prepare train and validation dataloaders...')
to_tuple = ["voxels", "images"]
val_batch_size = 1
split_by_node = lambda urls: urls
val_url = list(braceexpand.braceexpand(val_url))
val_data = wds.WebDataset(val_url, resampled=False, cache_dir=data_path, nodesplitter=split_by_node) \
    .decode("torch")\
    .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy") \
    .to_tuple(*to_tuple) \
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)

voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
num_voxels = voxels_per_subj.get(subj)

kwargs = {'hidden_dim': 1024, 'out_dim': feat_dim, 'num_latents': 256, 'sub': subj}

if fmri_encoder == 'brainxc':
    voxel2emb = BrainXC(**kwargs)
else:
    raise ValueError("The fmri encoder is not implemented.")
voxel2emb.to(device)

checkpoint = torch.load(tokenizer_path, map_location='cpu')
voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)

checkpoint = torch.load(encoder_path, map_location='cpu')
voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
voxel2emb.eval()

# inference: predict image features from fmri
print('inference: predict image features from fmri...')
emb_voxel_list, image_list = [], []
for val_i, (voxel, image) in enumerate(val_dl): 
    with torch.no_grad():
        with torch.cuda.amp.autocast():
           # repeat_index = val_i % 3
            # voxel = voxel[:,repeat_index].float()
            voxel = torch.mean(voxel, axis=1).float()
            
            emb_voxel = voxel2emb(voxel.to(device))
                
            emb_voxel_list.append(emb_voxel)
            if save_image:
                image_list.append(image) # for visualization

# assign image features to the predicted features from fmri
image_features = torch.cat(emb_voxel_list, dim=0)
print(f"image_features.shape: {image_features.shape}")

if save_image:
    image_list = torch.cat(image_list, dim=0)

# load llama with the fine-tuned shikra model
finetuned_llama = shikra_path # 'model_weights/shikra-7b' # shikra
tokenizer = LlamaTokenizer.from_pretrained(finetuned_llama, padding_side='left')
model = LlamaForCausalLM.from_pretrained(finetuned_llama)
model.to(device)

if feat_dim == 1024:
    # load mm_projector
    mm_projector = torch.nn.Linear(1024, 4096)
    mm_projector_weights = torch.load(adapter_path, map_location='cpu')
    if adapter_path == 'model_weights/mm_projector.bin':
        adjusted_state_dict = {k.split('.')[-1]: v for k, v in mm_projector_weights.items()}
        mm_projector.load_state_dict(adjusted_state_dict)
    else:
        mm_projector.load_state_dict(mm_projector_weights['model_state_dict'], strict=False)

    mm_projector.to("cuda:0")
    image_features = mm_projector(image_features.to(torch.float32))
    print(f"image_features.shape: {image_features.shape}")

# process prompt
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
user_image = " <im_start>" + "<im_patch>" * 256 + "<im_end> "

if '<image>' in prompt:
    user_prompt = prompt.replace('<image>', user_image)
else:
    user_prompt = prompt + user_image
input_text = system + user_prompt + " ASSISTANT:"

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)#.cuda()
inputs_embeds = model.model.embed_tokens(input_ids)

gen_kwargs = dict(
    use_cache=True,
    do_sample=False,
    pad_token_id=2, # tokenizer.pad_token_id,
    bos_token_id=1, # tokenizer.bos_token_id,
    eos_token_id=2, # tokenizer.eos_token_id,
    max_new_tokens=512,
)

cap_result = {}
for cur_image_idx in range(image_features.shape[0]):
    new_input_embeds = []
    for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
        cur_image_features = image_features[cur_image_idx]
        num_patches = cur_image_features.shape[0]
        image_start_tokens = torch.where(cur_input_ids == 32001)[0]
        for image_start_token_pos in image_start_tokens:
            cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
            num_patches = cur_image_features.shape[0]
            if cur_input_ids[image_start_token_pos + num_patches + 1] != 32002:
                raise ValueError("The image end token should follow the image start token.")
            
            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos + 1], cur_image_features,
                                                    cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
        new_input_embeds.append(cur_new_input_embeds)
    inputs_embeds = torch.stack(new_input_embeds, dim=0)

    st_time = time.time()
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            output_ids = model.generate(inputs_embeds=inputs_embeds.float(), **gen_kwargs)
    print(f"done generated in {time.time() - st_time} seconds")

    response = tokenizer.batch_decode(output_ids)[0]

    # print(f"input: {input_text}")
    print(f"response: {response.strip(' <s></s>')}")

    # save response in a txt file
    with open(os.path.join(result_dir, 'response.txt'), 'a') as f:
        f.write(f'response_{cur_image_idx}: \n')
        f.write(response + '\n')

    # save response in a json file
    bbox, caption = utils.extract_id_bbox_caption(response)
    cap_result[str(cur_image_idx)] = caption # {image_name: caption; ...}

    # save processed image (only for bbox tasks)
    if save_image:
        _, processed_image = postprocess(response, image=ToPILImage()(image_list[cur_image_idx]), width=5)
        if processed_image is not None:
            output_path = os.path.join(result_dir, f'{cur_image_idx}_prompt.png')
            processed_image.save(output_path)

with open(os.path.join(result_dir, 'fmricap.json'), 'w') as f:
    json.dump(cap_result, f)