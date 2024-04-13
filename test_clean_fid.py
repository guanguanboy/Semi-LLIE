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
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid_metric = pyiqa.create_metric('fid').to(device)
#fid_score = fid_metric('/data/liguanlin/codes/research_project/Semi-UIR/data/unlabeled_test/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0405/')
GT_path = '/data/liguanlin/codes/research_project/Semi-UIR/data/LOLv1/val/GT/'
result_path = '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0410_on_LOLv1_new/'
fid_score = fid_metric('/data/liguanlin/codes/research_project/Semi-UIR/data/LOLv1/val/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/ckpt_begin_0410_on_LOLv1_new/')
print('fid_score=', fid_score)


lpips_metric = pyiqa.create_metric('lpips').to(device)
niqe_metric = pyiqa.create_metric('niqe').to(device)
psnr_metric = pyiqa.create_metric('psnr',  color_space='ycbcr').to(device)
ssim_metric = pyiqa.create_metric('ssim', color_space='ycbcr').to(device)

GT_paths = []
enhanced_paths = []

for file in os.listdir(result_path):
    file_path = os.path.join(result_path, file)
    gt_path = os.path.join(GT_path, file)
    if os.path.isfile(file_path):
        enhanced_paths.append(file_path)
        GT_paths.append(gt_path)

count = 0

psnr_scores = []
ssim_scores = []
lpips_scores = []
mse_scores = []
niqe_scores = []

for i, enhanced_path in enumerate(tqdm(enhanced_paths)):
    GT_path = GT_paths[i]

    lpips_score = lpips_metric(GT_path, enhanced_path).cpu().numpy().item()
    #print('lpips_score=', lpips_score)
    lpips_scores.append(lpips_score)

    niqe_score = niqe_metric(enhanced_path).cpu().numpy().item()
    niqe_scores.append(niqe_score)

    psnr_score = psnr_metric(GT_path, enhanced_path).cpu().numpy().item()
    #print('psnr_score=', psnr_score)
    psnr_scores.append(psnr_score)

    ssim_score = ssim_metric(GT_path, enhanced_path).cpu().numpy().item()
    #print('psnr_score=', ssim_score)
    ssim_scores.append(ssim_score)


#使用skimage中的函数计算pnsr和ssim
image_size = 256
psnr_scores2 = []
ssim_scores2 = []
mse_scores2 = []

for i, enhanced_path in enumerate(tqdm(enhanced_paths)):
    GT_path = GT_paths[i]
    gt = Image.open(GT_path).convert('RGB')
    enhanced = Image.open(enhanced_path).convert('RGB')

    gt = tf.resize(gt,[image_size,image_size], interpolation=Image.BILINEAR)
    enhanced = tf.resize(enhanced,[image_size,image_size], interpolation=Image.BILINEAR)

    gt_np = np.array(gt, dtype=np.float32)
    enhanced_np = np.array(enhanced, dtype=np.float32)

    gt = tf.to_tensor(gt_np).unsqueeze(0).to(device)
    enhanced = tf.to_tensor(enhanced_np).unsqueeze(0).to(device)

    mse_score = mse(enhanced_np, gt_np)
    mse_scores2.append(mse_score)

    psnr_score = psnr(enhanced_np, gt_np, data_range=255)
    psnr_scores2.append(psnr_score)

    ssim_score = ssim(enhanced_np, gt_np, channel_axis=2,data_range=255, win_size=11)
    ssim_scores2.append(ssim_score)


def calculate_average(lst):
    total = sum(lst)
    average = total / len(lst)
    return average

# 示例用法
avg_psnr = calculate_average(psnr_scores)
print('avg psnr =', avg_psnr)

avg_ssim = calculate_average(ssim_scores)
print('avg ssim =', avg_ssim)

avg_lpips = calculate_average(lpips_scores)
print('avg lpips =', avg_lpips)

avg_niqe = calculate_average(niqe_scores)
print('avg niqe =', avg_niqe)

avg_psnr2 = calculate_average(psnr_scores2)
print('avg psnr2 =', avg_psnr)

avg_ssim2 = calculate_average(ssim_scores2)
print('avg ssim2 =', avg_ssim2)


avg_mse2 = calculate_average(mse_scores2)
print('avg mse =', avg_mse2)