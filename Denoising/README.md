# Gaussian Color Image Denoising

## Training
* Download training (DIV2K, WaterlooED, BSD400 and Flickr2K) and testing datasets, run
```
python download_data.py --data train-test --noise gaussian
``` 

* Generate image patches from full-resolution training images, run
```
python generate_patches_dfwb.py 
```

* Train SUTNet for gaussian color image denoising, run
```
python ./basicsr/train.py -opt ..\Denoising\Options\GaussianColorDenoising_SUTNetSigma15.yml
python ./basicsr/train.py -opt ..\Denoising\Options\GaussianColorDenoising_SUTNetSigma25.yml
python ./basicsr/train.py -opt ..\Denoising\Options\GaussianColorDenoising_SUTNetSigma50.yml
```

* Train lightweight SUTNet for gaussian color image denoising, run
```
python ./basicsr/train.py -opt ..\Denoising\Options\Light_GaussianColorDenoising_SUTNetSigma15.yml
python ./basicsr/train.py -opt ..\Denoising\Options\Light_GaussianColorDenoising_SUTNetSigma25.yml
python ./basicsr/train.py -opt ..\Denoising\Options\Light_GaussianColorDenoising_SUTNetSigma50.yml
```

## Evaluation
* Download the pre-trained [models](https://drive.google.com/drive/folders/14fDFI8g-WXxsK-GxZVfYh7fW_nx4Yvkn?usp=share_link) and place them in `./pretrained_models/`

* Download testsets (BSD68, Kodak24, McMaster and Urban100), run
```
python download_data.py --data test --noise gaussian
``` 

* To obtain denoised predictions using SUTNet, run
```
python test_gaussian_color_denoising.py --weights ./pretrained_models/gaussian_color_denoising --sigmas 15,25,50
```

* To obtain denoised predictions using lightweight SUTNet, run
```
python test_gaussian_color_denoising.py --weights ./pretrained_models/light_gaussian_color_denoising --sigmas 15,25,50
```

* To obtain PSNR and SSIM scores, run
```
python evaluate_gaussian_color_denoising.py --sigmas 15,25,50
```

# Real Image Denoising

## Training
* Download SIDD-Medium Dataset, run
```
python download_data.py --data train --noise real
``` 

* Generate image patches from full-resolution training images, run
```
python generate_patches_sidd.py 
```

* Train SUTNet for real image denoising, run
```
python ./basicsr/train.py -opt ..\Denoising\Options\RealDenoising_SUTNet.yml
```

* Train lightweight SUTNet for real image denoising, run
```
python ./basicsr/train.py -opt ..\Denoising\Options\Light_RealDenoising_SUTNet.yml
```

## Evaluation
* Download the pre-trained [models](https://drive.google.com/drive/folders/1fyaTwacjGCdDGPCvGZ4ka6hfP3GVIS6B?usp=share_link) and place them in `./pretrained_models/`

* Download SIDD Validation Dataset, run
```
python download_data.py --data test --dataset SIDD --noise real
``` 

* To obtain denoised predictions using SUTNet, run
```
python test_real_denoising_sidd.py --weights ./pretrained_models/real_denoising.pth 
```

* To obtain denoised predictions using lightweight SUTNet, run
```
python test_real_denoising_sidd.py --weights ./pretrained_models/light_real_denoising.pth 
```

* To obtain PSNR and SSIM scores, run this MATLAB script
```
evaluate_sidd.m
```