#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/01/15 11:47:04
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   single-subject training script
@usage   :  accelerate launch --num_processes=1 --num_machines=1 --gpu_ids='0' train.py \
                --data_path '/home/wx258/project/nsd_data' --fmri_encoder 'brainxs' \
                --model_save_path 'train_logs/demo_single_subject/sub01_dim1024' \
                --resume 'train_logs/demo_single_subject/sub01_dim1024/last.pth' 
'''

import os
import sys
import json
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import utils

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))
from model import BrainEncoder, BrainXS

import warnings
warnings.filterwarnings('ignore')

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# multi-GPU config
from accelerate import Accelerator
accelerator = Accelerator(split_batches=False, mixed_precision='fp16')  
print("PID of this process =", os.getpid())
print = accelerator.print

device = accelerator.device
print("device:", device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

# configurations
parser = argparse.ArgumentParser(description='Model Training Configuration')
parser.add_argument('--model_name', type=str, default='training_demo', help='name of model, used for ckpt saving')
parser.add_argument('--data_path', type=str, default='nsd_data', help='path to where NSD data is stored / where to download it to')
parser.add_argument('--model_save_path', type=str, default='', help='path to save results')
parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
parser.add_argument('--feat_dim', type=int, help='feature: 1024 (ViT) or 4096 (LLM)', default=1024)
parser.add_argument('--fmri_encoder', type=str, default='brainxs', help='type of brainnet', choices=['brainxs'])
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=240, help='number of epochs of training')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training')
parser.add_argument('--max_lr', type=float, default=3e-4)
parser.add_argument('--recon_loss', type=str, default='mse', choices=['mse', 'l1', 'huber', 'quantile'])
parser.add_argument('--use_image_aug', action=argparse.BooleanOptionalAction, default=True, help='whether to use image augmentation')
parser.add_argument('--plot_umap', action=argparse.BooleanOptionalAction, default=True, help='plot UMAP plots')
parser.add_argument('--lr_scheduler_type', type=str, default='cycle', choices=['cycle','linear'])
parser.add_argument('--ckpt_interval', type=int, default=5, help='save backup ckpt and reconstruct every x epochs')
parser.add_argument('--ckpt_saving', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--save_at_end', action=argparse.BooleanOptionalAction, default=False, help='if True, saves best.ckpt at end of training. \
        if False and ckpt_saving==True, save best.ckpt whenever epoch shows best validation score')
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# create output directory
if model_save_path:
    outdir = model_save_path
else:
    date = datetime.datetime.now().strftime("%y%m%d")
    outdir = os.path.abspath('./train_logs/{}_{}/sub{:02d}_dim{}'.format(model_name, date, subj, feat_dim))
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# save config in a json file
args_dict = vars(args)
with open(os.path.join(outdir, 'config.json'), 'w') as file:
    json.dump(args_dict, file, indent=4)

# with open(os.path.join(outdir, 'config.txt'), 'w') as file:
#     for key, value in args_dict.items():
#         file.write(f"{key}={value}\n")

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed, cudnn_deterministic=False)

# change learning rate based on number of devices
max_lr *= accelerator.num_processes
    
# change batch size based on number of devices if using multi-gpu
batch_size *= accelerator.num_processes

# change num_epochs based on number of devices if using multi-gpu
num_epochs *= accelerator.num_processes

if use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
    )

# prepare models and data loaders
print('\nprepare NSD webdataset data...')

train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"
val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
print(train_url, "\n", val_url)
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982

print('\nprepare train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size, 'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=300,
    cache_dir=data_path, 
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images"],
    subj=subj,
)

voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
num_voxels = voxels_per_subj.get(subj)

print(f'\ncreating brainencoder: {fmri_encoder}')
image2emb = BrainEncoder()
image2emb.to(device)

if fmri_encoder == 'brainxs':
    voxel2emb = BrainXS(in_dim=num_voxels, hidden_dim=1024, out_dim=feat_dim, num_latents=256)
else:
    raise ValueError("The fmri encoder is not implemented.")
voxel2emb.to(device)

print(f"start training from scratch")
    
print("\nparams of brainencoder")
if local_rank==0:
    utils.count_params(voxel2emb)

voxel2emb.requires_grad_(True)
image2emb.requires_grad_(False)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2emb.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2emb.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

global_batch_size = batch_size * num_devices
if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters = int(num_epochs*(num_train//global_batch_size)),
        last_epoch = -1
    )
elif lr_scheduler_type == 'cycle':
    total_steps = int(num_epochs*(num_train//global_batch_size))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )

def get_loss_func(recon_loss):
    loss_functions = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'quantile': lambda x, y: torch.quantile(torch.abs(x - y), 0.9)
    }
    if recon_loss not in loss_functions:
        raise ValueError(f"Unrecognized loss type: {recon_loss}")
    return loss_functions[recon_loss]
    
def save_ckpt(tag):    
    ckpt_path = outdir + f'/{tag}.pth'
    print(f'\nsaving {ckpt_path}', flush=True)
    unwrapped_model = accelerator.unwrap_model(voxel2emb)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
            }, ckpt_path)
    except:
        print("Couldn't save... moving on to prevent crashing.")
    del unwrapped_model
        
print("\nDone with model preparations")

# main loop for training
if resume:
    print(f"loading checkpoint from {resume}")
    checkpoint = torch.load(resume, map_location='cpu')
    voxel2emb.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    lrs = checkpoint['lrs']
    print(f"resuming from epoch {epoch}")
    best_val_loss = min(val_losses)
else:
    # epoch = 0 if not resume else epoch + 1
    epoch = 0
    losses, val_losses, lrs = [], [], []
    best_val_loss = 1e9
    print(f"starting from scratch")

voxel2emb, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
voxel2emb, optimizer, train_dl, val_dl, lr_scheduler
)

if feat_dim == 4096:
    # load mm_projector
    mm_projector = torch.nn.Linear(1024, 4096)
    mm_projector_weights = torch.load('model_weights/mm_projector.bin', map_location='cpu')
    mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
    mm_projector.to("cuda:0")

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), ncols=120, disable=(local_rank!=0))

loss_fn = get_loss_func(recon_loss)

for epoch in progress_bar:
    voxel2emb.train()

    loss_recon_sum = 0.
    val_loss_recon_sum = 0.

    for train_i, (voxel, image) in enumerate(train_dl):
        with torch.cuda.amp.autocast():
            optimizer.zero_grad()

            repeat_index = train_i % 3
                
            voxel = voxel[:,repeat_index].float()

            emb_voxel = voxel2emb(voxel)

            if use_image_aug:
                image = img_augment(image)
            emb_image = image2emb.encode_image(image, 'image') # torch.Size([16, 256, 1024])
            if feat_dim == 4096:
                emb_image = mm_projector(emb_image.to(torch.float16))

            loss_recon = loss_fn(emb_voxel, emb_image)

            loss_recon_sum += loss_recon.item()
            loss = loss_recon 
            utils.check_loss(loss)
            
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    voxel2emb.eval()
    for val_i, (voxel, image) in enumerate(val_dl): 
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # repeat_index = val_i % 3
                # voxel = voxel[:,repeat_index].float()
                voxel = torch.mean(voxel, axis=1).float()
                emb_voxel = voxel2emb(voxel)

                emb_image = image2emb.encode_image(image, 'image')
                if feat_dim == 4096:
                    emb_image = mm_projector(emb_image.to(torch.float16))

                val_loss_recon = loss_fn(emb_voxel, emb_image)
                
                val_loss_recon_sum += val_loss_recon.item()
                val_loss = val_loss_recon
                utils.check_loss(val_loss)
                val_losses.append(val_loss.item())
                
    if local_rank==0:        
        if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i+1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
            
        logs = {"train/loss": np.mean(losses[-(train_i+1):]),
            "val/loss": np.mean(val_losses[-(val_i+1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "val/num_steps": len(val_losses),
            "train/loss_recon": loss_recon_sum / (train_i + 1),
            "val/loss_recon": val_loss_recon_sum / (val_i + 1)}
        progress_bar.set_postfix(**logs)

        # save logs
        with open(os.path.join(outdir, 'logs.txt'), 'a') as f:
            f.write(json.dumps({'epoch': epoch, **logs}) + '\n')

        # save model checkpoint
        save_ckpt(f'last')
        if epoch % ckpt_interval == 0:
            save_ckpt(f'last_backup')
            if plot_umap:
                import umap # pip install umap-learn
                print('umap plotting...')
                combined = np.concatenate((emb_image.flatten(1).detach().cpu().numpy(),
                                            emb_voxel.flatten(1).detach().cpu().numpy()), axis=0)
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(combined)

                colors = np.array([[0, 0, 1, .5] for i in range(len(emb_image))])
                colors = np.concatenate((colors, np.array([[0, 1, 0, .5] for i in range(len(emb_voxel))])))

                fig = plt.figure(figsize=(5, 5))
                plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=colors)
                plt.savefig(os.path.join(outdir, f'umap-val-epoch{epoch:03d}.png'))            
           
    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()

# training and validation loop ends here
# draw and save plots of training and validation losses

def plot_and_save(data, label, filename, outdir, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.plot(data, label=label)
    plt.legend()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()  # Close the plot to free up memory

# Plot and save train loss
plot_and_save(losses, 'train loss', 'loss_train.png', outdir)
# Plot and save validation loss
plot_and_save(val_losses, 'val loss', 'loss_val.png', outdir)
# Plot and save learning rate
plot_and_save(lrs, 'lr', 'lr.png', outdir)