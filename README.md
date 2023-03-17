# Challenge_ZhongKeXingTu5_Sea_Ice

This repository contains our code in the  Gaofen Challenge. The task of this challenge is to extract sea ice targets from a large range of sea areas in a fine pixel-by-pixel manner.

## Dataset

Ocean 1 optical data with a resolution of 50m. The scene covers the surrounding areas of the Bohai Sea in China. The image size varies from 512 to 2048 pixels. Remote sensing images are in tif format and contain R-G-B channels; The labeling file is in single-channel png format with sea ice pixel values of 255 and background pixel values of 0.

We cut images larger than 512 pixels into several 512-pixel images, and store the cut data set in csv.

```python
data
|--train.csv
|--val.csv
trainData
|--gt
|  |--1.png
|  |--2.png
|   ......
|--image
|  |--1.tif
|  |--2.tif
|   ......
```



## Train

We randomly split the provided images into the training and test sets with a ratio of 9: 1. For data augmentation, HorizoalFlip and RandomRotate90 are adopted and the images are normalized with the mean and standard deviation values of 127.5 and 31.875, respectively.

## Model

The time during the finals is an important evaluation index. Unet architecture is adopted in the whole experiment. In order to reduce the number of parameters and speed up, the number of coding and decoding blocks is set to four. We use [timm-efficientnet-lite3]([[1905.11946v5\] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (arxiv.org)](https://arxiv.org/abs/1905.11946v5)) as our backbone feature extraction. Compared with the same series of timm-efficientnet-lite4, the speed is faster. Although the accuracy has decreased, the difference is not significant.

## Test