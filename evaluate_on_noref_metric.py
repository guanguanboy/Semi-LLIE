"""
from cleanfid import fid
score = fid.compute_fid('/data/liguanlin/codes/research_project/Semi-UIR/data/unlabeled_test/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0316/', mode="clean", num_workers=1, batch_size=16)
print(score)
"""
import os
import pyiqa
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import pytorch_ssim
import torchvision.transforms.functional as tf
from PIL import Image

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#fid_score = fid_metric('/data/liguanlin/codes/research_project/Semi-UIR/data/unlabeled_test/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0405/')
result_path = '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0410_on_LOLv1_new/VV/'

#CLIPIQA
#MUSIQ

niqe_metric = pyiqa.create_metric('niqe').to(device)
clipiqa_metric = pyiqa.create_metric('clipiqa').to(device)
musiq_metric = pyiqa.create_metric('musiq').to(device)

enhanced_paths = []

for file in os.listdir(result_path):
    file_path = os.path.join(result_path, file)
    if os.path.isfile(file_path):
        enhanced_paths.append(file_path)

niqe_scores = []
clipiqa_scores = []
musiq_scores = []

for i, enhanced_path in enumerate(tqdm(enhanced_paths)):

    niqe_score = niqe_metric(enhanced_path).cpu().numpy().item()
    niqe_scores.append(niqe_score)

    clipiqa_score = clipiqa_metric(enhanced_path).cpu().numpy().item()
    clipiqa_scores.append(clipiqa_score)
    
    musiq_score = musiq_metric(enhanced_path).cpu().numpy().item()
    musiq_scores.append(musiq_score)

def calculate_average(lst):
    total = sum(lst)
    average = total / len(lst)
    return average

# 示例用法

avg_niqe = calculate_average(niqe_scores)
print('avg niqe =', avg_niqe)

avg_clipiqa = calculate_average(clipiqa_scores)
print('avg clipiqa =', avg_clipiqa)

avg_musiq = calculate_average(musiq_scores)
print('avg musiq =', avg_musiq)