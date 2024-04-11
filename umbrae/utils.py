import os
import re
import random
import math
import json
import requests
import braceexpand
from PIL import Image
import numpy as np

import webdataset as wds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_id_bbox_caption(input_str):
    bbox = re.findall(r'(\w+)\[(.*?)\]', input_str)
    caption = re.sub(r'\[\d.*?\]', '', input_str).strip(" <s></s>")
    return bbox, caption

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

def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def get_huggingface_urls(commit='main', subj=1):
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/"
    train_url = base_url + commit + f"/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar"
    val_url = base_url + commit + f"/webdataset_avg_split/val/val_subj0{subj}_0.tar"
    test_url = base_url + commit + f"/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    return train_url, val_url, test_url
    
def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
    to_tuple=["voxels", "images", "trial"],
    subj=1,
    data_ratio=1.0,
):
    print("Getting dataloaders...")
    assert image_var == 'images'
    
    def my_split_by_node(urls):
        return urls
    
    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))
    if not os.path.exists(train_url[0]):
        # we will default to downloading from huggingface urls if data_path does not exist
        print("downloading NSD from huggingface...")
        os.makedirs(cache_dir, exist_ok=True)
        
        train_url, val_url, test_url = get_huggingface_urls("main", subj)
        train_url = list(braceexpand.braceexpand(train_url))
        val_url = list(braceexpand.braceexpand(val_url))
        test_url = list(braceexpand.braceexpand(test_url))
        
        from tqdm import tqdm
        for url in tqdm(train_url):
            destination = cache_dir + "/webdataset_avg_split/train/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(val_url):
            destination = cache_dir + "/webdataset_avg_split/val/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(test_url):
            destination = cache_dir + "/webdataset_avg_split/test/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_train", num_train)
    print("global_batch_size", global_batch_size)
    print("batch_size", batch_size)
    print("num_workers", num_workers)
    print("num_batches", num_batches)
    print("num_worker_batches", num_worker_batches)
    
    num_samples = int(num_train * data_ratio)
    train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .slice(num_samples)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)
    
    train_dl = DataLoader(train_data, batch_size=None, num_workers=1, shuffle=False)

    # validation (no shuffling, should be deterministic)  
    num_batches = math.floor(num_val / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_val", num_val)
    print("val_num_batches", num_batches)
    print("val_batch_size", val_batch_size)
    
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=False)
    
    val_dl = DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)

    return train_dl, val_dl, num_train, num_val

# process the bounding boxes
def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box

def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def draw_bounding_boxes(
        image,
        boxes,
        **kwargs,
):
    if isinstance(image, Image.Image):
        image = transforms.PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, **kwargs)

colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

def extract_boxes(string):
    ret = []
    for bboxes_str in pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret
    
def postprocess(text, image, width=8):
    if image is None:
        return text, None

    image = expand2square(image)

    extract_pred = extract_boxes(text)
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)
    if not boxes_to_draw:
        return text, None
    res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=width)
    res = transforms.ToPILImage()(res)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color = colors[idx % len(colors)]
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res