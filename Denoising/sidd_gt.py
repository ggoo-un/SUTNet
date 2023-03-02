import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.real_restormer_arch import Real_Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='F:/results/real\SUTNet_39.88\mat', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='F:\Restormer-main/result/Real_Denoising/SUTNet_mat2img/', type=str, help='Directory for results')

args, unknown = parser.parse_known_args()

os.makedirs(args.result_dir, exist_ok=True)

# Process data
filepath = os.path.join(args.input_dir, 'Idenoised.mat')
img = sio.loadmat(filepath)
GT = np.float32(np.array(img['Idenoised']))
# GT /=255.
restored = np.zeros_like(GT)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(GT[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()

            # noisy_patch = torch.clamp(noisy_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            noisy_patch = noisy_patch.cpu().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = noisy_patch


            save_file = os.path.join(args.result_dir, '%04d_%02d.png'%(i+1,k+1))
            utils.save_img(save_file, img_as_ubyte(noisy_patch))

