# MDCN_PyTorch
### This repository is an official PyTorch implementation of “MDCN".

### This paper has been submitted to TCSVT.

### We will publish the paper and authors information after the paper is accepted.

### More detailed ablation analysis experiments will be publish after the paper is accepted.

### All reconstructed images will be provided soon.


## Prerequisites:
1. Python 3.6
2. PyTorch >= 0.4.0
3. numpy
4. skimage
5. imageio
6. matplotlib
7. tqdm


## Dataset

We used DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>.

Extract the file and put it into the Train/dataset.

Only DIV2K is used as the training dataset, and Flickr2K is not used as the training dataset !!!


## Performance

Our MDCN is trained on RGB. Following previous works, we only reported PSNR/SSIM on the Y channel in YCbCr space.


<p align="center">
  <img src="images/Results.png" width="700px"> 
</p>

<p align="center">
  <img src="images/Visual.png" width="700px" > 
</p>

<p align="center">
  <img src="images/MSRN.png" width="500px"> 
</p>



## Convergence Analyses

MDCN x2 on DIV2K training dataset  and  validation dataset.

<p align="center">
<img src="images/loss_L1_x2.png" width="400px" height="300px"/> <img src="images/test_DIV2K_x2.png" width="400px" height="300px"/> 
</p>

MDCN x3 on DIV2K training dataset  and  validation dataset.

<p align="center">
<img src="images/loss_L1_x3.png" width="400px" height="300px"/> <img src="images/test_DIV2K_x3.png" width="400px" height="300px"/> 
</p>

MDCN x4 on DIV2K training dataset  and  validation dataset.

<p align="center">
<img src="images/loss_L1_x4.png" width="400px" height="300px"/> <img src="images/test_DIV2K_x4.png" width="400px" height="300px"/> 
</p>

