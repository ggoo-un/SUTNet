# SUTNet: Image Restoration Network Using Swin-Unet Transformer to Improve Camera Image Quality


[![paper](https://img.shields.io/badge/Master%20of%20Science-Thesis-brightgreen)](https://drive.google.com/file/d/1uPaf3Sy0P4k16cOYm4-hncsLPPDtoO4j/view?usp=share_link)
[![slides](https://img.shields.io/badge/Presentation-Slides-red.svg)](https://drive.google.com/file/d/1S4OndFWO7pbjpGEMq7Xe-v-7bXGcghQ8/view?usp=share_link)
![python](https://img.shields.io/badge/Python-FFE135?style=flat&logo=Python&logoColor=#3776AB)
![pytorch](https://img.shields.io/badge/PyTorch-white?style=flat&logo=PyTorch&logoColor=#EE4C2C)

<hr />

> **Abstract:** 
> *In this paper, deep learning-based denoising network (Gaussian color image denoising and real image denoising) 
> and motion deblurring network are dealt with for the most serious image degradation problems. The proposed model is implemented 
> with the Swin-Unet transformer as the basic structure. The structure is constructed by grafting the Swin (Shifted window) 
> Transformer block, which has relatively little amount of computation in Transformer and can learn the context of image pixels, 
> to the U-Net structure that is widely used in the medical field and is beneficial for improving contour details. 
> In the Swin Transformer block of the proposed neural network, the Windows Multi-head Dconv Attention (WMDA) module replaces 
> the existing Multi-head Self Attention (MSA), and the Locally-enhanced Gated Dconv feed-forward network (LeGD) module replaces 
> the existing Feed Forward Network (FNN). Through this, the proposed model makes it effective for image restoration by well 
> understanding the context between adjacent pixels and the context between channels and preserving image details.* 

<hr />

The Master of Science thesis can be found [here](https://drive.google.com/drive/folders/1IVTJJdkdZaFbWe5Ohn0aMlMQVuy7A00Z?usp=share_link).

## Network Architecture
![proposed_model](img/proposed_model.jpg)

## Training and Evaluation

The used training and testing datasets and visual results can be downloaded as follows:

<table>
  <tr>
    <th align="center">Task</th>
    <th align="center">Training Datasets</th>
    <th align="center">Testing Datasets</th>
    <th align="center">SUTNet's Pre-trained Models</th>
    <th align="center">SUTNet's Visual Results</th>
  </tr>
  <tr>
    <td align="center">Gaussian Color Image Denoising</td>
    <td align="center">DIV2K, WaterlooED, BSD400 and Flickr2K</td>
    <td align="center">BSD68, Kodak24, McMaster and Urban100</td>
    <td align="center"><a href="https://drive.google.com/drive/folders/14fDFI8g-WXxsK-GxZVfYh7fW_nx4Yvkn?usp=share_link">Download</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1imVUt6VVLWVxAp9LTNVGV-codyVB2Ox_?usp=share_link">Here</a></td>
  </tr>
  <tr>
    <td align="center">Real Image Denoising</td>
    <td align="center">SIDD-Medium Dataset</td>
    <td align="center">SIDD Validation Dataset</td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1fyaTwacjGCdDGPCvGZ4ka6hfP3GVIS6B?usp=share_link">Download</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1Y7DFmFIlyYe8ur7fzC-_8XQ6Qj0syTzf?usp=share_link">Here</a></td>
  </tr>
  <tr>
    <td align="center">Motion Deblurring</td>
    <td align="center" colspan="2">GoPro Dataset</td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1zHXWOa7DzzfXxx8Kn8UOS3ybkmVKNUcL?usp=share_link">Download</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1LU1eoWx8asTX4tHU9BdMaWGKWMHZmm79?usp=share_link">Here</a></td>
  </tr>
</table>

## Instructions

<table>
  <tr>
    <th align="center">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
  </tr>
  <tr>
    <td align="center">Gaussian Color Image Denoising</td>
    <td align="center"><a href="Denoising/README.md#training">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation">Link</a></td>
  </tr>
  <tr>
    <td align="center">Real Image Denoising</td>
    <td align="center"><a href="Denoising/README.md#training-1">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation-1">Link</a></td>
  </tr>
  <tr>
    <td align="center">Motion Deblurring</td>
    <td align="center"><a href="Motion_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Motion_Deblurring/README.md#evaluation">Link</a></td>
  </tr>
</table>


## Results

Experiments are performed for different image restoration tasks including, gaussian color image denoising, real image denoising, and motion deblurring. Detailed results can be found in the paper.

### PSNR and SSIM scores

<details>
<summary><strong>Gaussian Color Image Denoising</strong></summary>
<br>
<img src="img/color_image_denoising_psnr.jpg" alt="color_image_denoising_psnr.jpg">
<img src="img/color_image_denoising_ssim.jpg" alt="color_image_denoising_ssim.jpg">
</details>

<details>
<summary><strong>Real Image Denoisng</strong></summary>
<br>
<img src="img/real_image_denoising_psnr_ssim.jpg" alt="real_image_denoising_psnr_ssim.jpg" width="480">
</details>

<details>
<summary><strong>Motion Deblurring</strong></summary>
<br>
<img src="img/motion_deblurring_psnr_ssim.jpg" alt="real_image_denoising_psnr_ssim.jpg" width="480">
</details>

### Visual results

<details>
<summary><strong>Gaussian Color Image Denoising</strong></summary>
<br>

- Gaussian color image denoising results of Urban100 with noise level 50.

<img src="img/color_image_denoising_result-1.jpg" alt="color_image_denoising_result-1.jpg">
<img src="img/color_image_denoising_result-2.jpg" alt="color_image_denoising_result-2.jpg">
</details>

<details>
<summary><strong>Real Image Denoisng</strong></summary>
<br>

- Real image denoising results of SIDD validation dataset.

<img src="img/real_image_denoising_result-1.jpg" alt="real_image_denoising_result-1.jpg">
<img src="img/real_image_denoising_result-2.jpg" alt="real_image_denoising_result-2.jpg">
</details>

<details>
<summary><strong>Motion Deblurring</strong></summary>
<br>

- Image motion deblurring results of GoPro dataset.

<img src="img/motion_deblurring_result.jpg" alt="motion_deblurring_result.jpg">

</details>

### Ablation study

<img src="img/ablation_study.jpg" alt="ablation_study.jpg" width="480">

The contribution of each component is analyzed through an ablation study to verify the performance of each module in the model. 
It demonstrated that the proposed model extracts the contextual information within the image and effectively improves the image quality.

Pre-trained models of SUTNet can be downloaded [here](#training-and-evaluation).

## Contact
Should you have any question, please contact gooni0906@gmail.com.
