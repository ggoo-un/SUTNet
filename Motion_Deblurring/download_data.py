## Download training and testing data for single-image motion deblurring task
import os
import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='test', help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='GoPro', help='all, GoPro, HIDE, RealBlur_R, RealBlur_J')
args = parser.parse_args()

### Google drive IDs ######
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'      ## https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing
GoPro_test  = '1k6DTSHu4saUgrGTYkkZXTptILyG9RRll'      ## https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/view?usp=sharing

dataset = args.dataset

for data in args.data.split('-'):
    if data == 'train':
        print('GoPro Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
        gdown.download(id=GoPro_train, output='Datasets/Downloads/train.zip', quiet=False)
        os.system(f'gdrive download {GoPro_train} --path Datasets/Downloads/')
        print('Extracting GoPro data...')
        shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
        os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'GoPro'))
        os.remove('Datasets/Downloads/train.zip')

    if data == 'test':
        print('GoPro Testing Data!')
        gdown.download(id=GoPro_test, output='Datasets/test.zip', quiet=False)
        os.system(f'gdrive download {GoPro_test} --path Datasets/')
        print('Extracting GoPro Data...')
        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
        os.remove('Datasets/test.zip')



# print('Download completed successfully!')
