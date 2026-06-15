# FracMix
FracMix:A Fractional Fourier Based Augmentationfor Generalizable Person Re-Identification.
# [IEEE Signal Processing Letters 2026] FracMix:A Fractional Fourier Based Augmentationfor Generalizable Person Re-Identification
## Paper Link
[FracMix: A Fractional Fourier Based Augmentation for Generalizable Person Re-Identification](https://ieeexplore.ieee.org/abstract/document/11503102)
## Framework
![image](https://github.com/HuidiXie/FracMix/blob/main/figures/net.png)
## Requirements
### Installation
we use /torch 1.13.1 /torchvision 0.14.1 /cuda 12.2 /four 24G RTX 4090 for running.
### Prepare Datasets
The model is run on Market-1501, MSMT17_V1, CUHK03 and CUHK-SYSU.</br>
Unzip all datasets and ensure the file structure is as follow:</br>
```
data
├── market1501
│    └── Market-1501-v15.09.15
│        └── images ..
├── msmt17v1
│    └── MSMT17_V1
│        └── images ..
├── cuhk03
│    └── cuhk03_release
│        └── images ..
└── cuhk_sysu
     └── croppped_images
         └── images ..
```
## Run
```
ARCH=ViT-B/16

Three-source domains(Protocol-1)
SRC1/SRC2/SRC3=market1501/cuhk03/msmt17v1/cuhk_sysu
TARGET=market1501/cuhk03/msmt17v1

Single-source domain(Protocol-2)
SRC1=market1501/msmt17v1
TARGET=msmt17v1/market1501

# train
CUDA_VISIBLE_DEVICES=1,2,3,4 python main_vit.py \
--dataset_src1 msmt17v1 -d market1501 \
-b 128 --test-batch-size 256 --height 256 --width 128 --num-instances 8 \
-a vit_base --BNNeck \
--epochs 40 --iters 400 --logs-dir log/1

CUDA_VISIBLE_DEVICES=1,2,3,4 python main_vit.py \
-d msmt17v1 --dataset_src1 market1501 --dataset_src2 cuhk_sysu --dataset_src3 cuhk03 --multi_source \
-b 128 --test-batch-size 256 --height 256 --width 128 --num-instances 8 \
-a vit_base --BNNeck \
--epochs 60 --iters 400 --logs-dir log/1

```

## Results
### Protocol-1</br>
![image](https://github.com/HuidiXie/FracMix/blob/main/figures/sota1.png)
### Protocol-2</br>
![image](https://github.com/HuidiXie/FracMix/blob/main/figures/sota2.png)
## Contact
If you have any question, please feel free to contact us.</br>
E-mail: [jierujia@sxu.edu.cn](mailto:jierujia@sxu.edu.cn)
