# Motion Deblurring

## Training
* Download GoPro training and testing datasets, run
```
python download_data.py --data train-test
``` 

* Generate image patches from full-resolution training images, run
```
python generate_patches_gopro.py
```

* Train SUTNet for motion deblurring, run
```
python ./basicsr/train.py -opt ..\Motion_Deblurring\Options\Deblurring_SUTNet.yml
```

* Train lightweight SUTNet for motion deblurring, run
```
python ./basicsr/train.py -opt ..\Motion_Deblurring\Options\Light_Deblurring_SUTNet.yml
```

## Evaluation
* Download the pre-trained [models](https://drive.google.com/drive/folders/1zHXWOa7DzzfXxx8Kn8UOS3ybkmVKNUcL?usp=share_link) and place them in `./pretrained_models/`

* Download GoPro testing dataset, run
```
python download_data.py --data test
``` 

* To obtain deblurred results using SUTNet, run
```
python test.py --dataset GoPro --weights ./pretrained_models/motion_deblurring.pth
```

* To obtain deblurred results using lightweight SUTNet, run
```
python test.py --dataset GoPro --weights ./pretrained_models/light_motion_deblurring.pth
```

* To obtain PSNR and SSIM scores, run this MATLAB script
```
evaluate_gopro.m
```