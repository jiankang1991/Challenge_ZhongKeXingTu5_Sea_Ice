# Challenge_ZhongKeXingTu5_Sea_Ice

This repository contains our code in the  Gaofen Challenge. The task of this challenge is to extract sea ice targets from a large range of sea areas in a fine pixel-by-pixel manner.

## Dataset
Ocean 1 optical data with a resolution of 50m. The scene covers the surrounding areas of the Bohai Sea in China. The image size varies from 512 to 2048 pixels. Remote sensing images are in tif format and contain R-G-B channels; The labeling file is in single-channel png format with sea ice pixel values of 255 and background pixel values of 0.

We cut images larger than 512 pixels into several 512-pixel images, and store the cut data set in csv.

```html
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

## Docker

Place the files in the same folder as the Dockerfile, and create a mirror from the Dockerfile.

```
docker build -t zkxt21/zkxt:版本号 .
```

See if the container works properly.

```
docker run --gpus all -v ./data/:/input_path --shm-size=2g -it zkxt21/zkxt:版本号 bash
```

Then run the command

```
python test.py
```

## Train

We randomly split the provided images into the training and test sets with a ratio of 9: 1. For data augmentation, HorizoalFlip and RandomRotate90 are adopted and the images are normalized with the mean and standard deviation values of 127.5 and 31.875, respectively.

For the binary segmentation task, commonly exploited losses in the literature are BCE, Dice, or their combinations. Compared to the pixel-based loss, i.e., BCE, Dice loss is more emphasized on learning precise region predictions. Most of the images in the training ground scene do not contain sea ice, causing the problem of gradients disappearing when dice are lost to the training network. We replace the last layer activated by the sigmoid function with the one activated by the softmax function and adopt the classwise Dice loss rather than the normal Dice loss for binary segmentation.

## Model

The time during the finals is an important evaluation index. A segmentation network with a U-shape structure is proposed which can simultaneously speed up the inference and preserve the detailed spatial information of sea ices. Differently with other methods, the 5th encoding blocks of the pretrained networks are omitted here, with the consideration of the balance between the computational cost and segmentation performance. 

In order to improve the efficiency, we adopt the residual learning scheme including depthwise and pointwise convolutions. In addition, the concurrent spatial and channel squeeze and excitation ([SCSE](https://arxiv.org/abs/1803.02579)) block is integrated to refine the features both along the spatial and channel dimensions. 

We use [timm-efficientnet-lite3](https://arxiv.org/pdf/1905.11946v5.pdf) as our backbone feature extraction. Compared with the same series of timm-efficientnet-lite4, the speed is faster. Although the accuracy has decreased, the difference is not significant.

## Test

In order to achieve the purpose of improving the reasoning speed without significantly reducing the accuracy, we have adopted a number of strategies:

1、Multi-thread image saving

2、Using defaultdict() to speed up dictionary extraction

3、Simplify weights saved during training and reduce time to read weights

4、Using [dali ](https://developer.nvidia.com/zh-cn/dali)to speed up picture reading

5、Torch.cuda.amp is used to automatically mix the precision in reasoning, which saves memory and speeds up reasoning.

## Citing this work

```html
@article{Kang2022DecodingTP,
  title={Decoding the Partial Pretrained Networks for Sea-Ice Segmentation of 2021 Gaofen Challenge},
  author={Jian Kang and Fengyu Tong and Xiang Ding and Sijiang Li and Ruoxin Zhu and Yan Huang and Yusheng Xu and Rub{\'e}n Fern{\'a}ndez-Beltran},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2022},
  volume={15},
  pages={4521-4530}
}
```