import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import utils


def proc(filename):
    tar, prd = filename
    tar_img = utils.load_img(tar)
    prd_img = utils.load_img(prd)

    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM


# datasets = ['CBSD68', 'Kodak', 'McMaster','Urban100']
datasets = ['SIDD_gt']

for dataset in datasets:

    gt_path = os.path.join('F:\Restormer-main/results', 'real', dataset)
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
    assert len(gt_list) != 0, "Target files not found"


    # file_path = os.path.join('./result', 'Gaussian_Color_Denoising', args.model_type, dataset, str(sigma_test))
    file_path = os.path.join('F:\Restormer-main/results/real', 'AINDNet_real')
    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim = [], []
    img_files = [(i, j) for i, j in zip(gt_list, path_list)]

    for filename, PSNR_SSIM in zip(img_files, map(proc, img_files)):
        psnr.append(PSNR_SSIM[0])
        ssim.append(PSNR_SSIM[1])

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)

    # print('For {:s} dataset Noise Level {:d} PSNR: {:f}\n'.format(dataset, sigma_test, avg_psnr))
    print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
