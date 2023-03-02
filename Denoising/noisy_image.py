import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/icsp/nvme1/alskyo/jg/SUTNet/')

from basicsr.models.archs.new_restormer_arch import NRestormer
from basicsr.models.archs.real_restormer_arch import Real_Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx


parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='F:\Restormer-main\Denoising\Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./result/Gaussian_Color_Noisy/', type=str, help='Directory for results')
parser.add_argument('--sigmas', default='15', type=str, help='Sigma values')

args, unknown = parser.parse_known_args()


sigmas = np.int_(args.sigmas.split(','))

# datasets = ['CBSD68', 'Kodak', 'McMaster','Urban100']
datasets = ['CBSD68', 'Kodak']

for sigma_test in sigmas:

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma_test))
        os.makedirs(result_dir_tmp, exist_ok=True)

        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img = np.float32(utils.load_img(file_))/255.

                np.random.seed(seed=0)  # for reproducibility
                img += np.random.normal(0, sigma_test/255., img.shape)

                img = torch.from_numpy(img).permute(2,0,1)
                input_ = img.unsqueeze(0).cuda()

                noisy = input_.cpu()

                noisy = torch.clamp(noisy,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                utils.save_img(save_file, img_as_ubyte(noisy))
