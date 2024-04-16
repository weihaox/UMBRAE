#!/bin/bash
# ------------------------------------------------------------------
# @File    :   download_checkpoint.sh
# @Time    :   2024/03/16 17:30:00
# @Author  :   Weihao Xia (xiawh3@outlook.com)
# @Version :   1.0
# @Desc    :   download Checkpoints from Hugging Face
# ------------------------------------------------------------------

python -c  'from huggingface_hub import snapshot_download; snapshot_download(repo_id="weihaox/brainx", repo_type="dataset", local_dir="./", ignore_patterns=["all_images.pt", ".gitattributes"])'