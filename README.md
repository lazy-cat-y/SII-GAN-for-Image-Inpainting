<!-- ---------------------------------------------------- -->
## Introduction 
Despite some promising results, it remains challenging for existing image inpainting approaches to fill in large missing regions in high resolution images (e.g., 512x512). We analyze that the difﬁculties mainly drive from simultaneously inferring missing contents and synthesizing fine-grained textures for a extremely large missing region. 
We propose a GAN-based model that improves performance by,
1) **Enhancing context reasoning by BDC Block in the generator.** The BDC blocks aggregate contextual transformations with different receptive fields, allowing to capture both informative distant contexts and rich patterns of interest for context reasoning. 
2) **Enhancing texture synthesis by SoftGAN in the discriminator.**  We improve the training of the discriminator by a tailored mask-prediction task. The enhanced discriminator is optimized to distinguish the detailed appearance of real and synthesized patches, which can in turn facilitate the generator to synthesize more realistic textures.


<!-- -------------------------------- -->
## Prerequisites 
* python 3.8.8
* [pytorch](https://pytorch.org/) (tested on Release 1.8.1)

<!-- --------------------------------- -->
## Datasets 

1. download images and masks
2. specify the path to training data by `--dir_image` and `--dir_mask`.

<!-- -------------------------------------------------------- -->
## Getting Started

1. Training:
    * Our codes are built upon distributed training with Pytorch.  
    * Run 
    ```
    cd src 
    python train.py  
    ```
2. Resume training:
    ```
    cd src
    python train.py --resume 
    ```
3. Testing:
    ```
    cd src 
    python test.py --pre_train [path to pretrained model] 
    ```
4. Evaluating:
    ```
    cd src 
    python eval.py --real_dir [ground truths] --fake_dir [inpainting results] --metric mae psnr ssim fid
    ```