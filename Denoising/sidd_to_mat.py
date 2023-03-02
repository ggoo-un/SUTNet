import glob

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import utils


from skimage import img_as_ubyte

import scipy.io as sio

import cv2


parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='F:\Restormer-main/results/real\SIDD_input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='F:\Restormer-main/results/real/SIDD_input_mat', type=str, help='Directory for results')

args, unknown = parser.parse_known_args()

os.makedirs(args.result_dir, exist_ok=True)

inputpath = glob.glob(os.path.join(args.input_dir, '*.png'))

# Process data
filepath = os.path.join('./Datasets/test/SIDD/', 'ValidationGtBlocksSrgb.mat')
img = sio.loadmat(filepath)
GT = np.float32(np.array(img['ValidationGtBlocksSrgb']))
GT /=255.
restored = np.zeros_like(GT)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            # GT_patch = torch.from_numpy(GT[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            # GT_patch = torch.clamp(GT_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored_patch = cv2.imread(inputpath[32*i+k])
            restored_patch = cv2.cvtColor(restored_patch, cv2.COLOR_BGR2RGB) / 255.

            restored[i,k,:,:,:] = restored_patch


# save denoised data
sio.savemat(os.path.join(args.result_dir, 'Idenoised.mat'), {"Idenoised": restored,})