**简体中文 | [English](./README.md)**

[![license](https://img.shields.io/github/license/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/blob/master/LICENSE) [![forks](https://img.shields.io/github/forks/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/members) [![stars](https://img.shields.io/github/stars/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/stargazers) ![python3](https://img.shields.io/badge/language-Python3-14274E?style=plastic&logo=python) ![anaconda3](https://img.shields.io/badge/environment-Anaconda3-394867?style=plastic&logo=anaconda) ![pytorch](https://img.shields.io/badge/framework-Pytorch-394867?style=plastic&logo=pytorch)

---

# About The Project

This project is to train semantic segmentation with supervised learning (Deeplabv3) and semi-supervised learning (NaturalHistoryMeuseum(NHM) semantic segmentation) for herbarium sheets and microscopy slides.

Training datasets are provided by: NMWHS, MNHNS, MIXSETHS.

-   The training datasets can be access here: [Download Link](https://zenodo.org/record/4448186)
-   In order to get more ideal result, I trained combination of original training datasets(2000 images) with small part of Wyoming's datasets(200 images) by using [labelme](https://anaconda.org/conda-forge/labelme). More details see in my code.

Prediction datasets are provided by Wyoming University.

-   The datasets used for predict can be accessed here: [Download]()

> **_If you like this project or it helps you in some way, don't forget to star._** :star:

## Table of content:

1. [NHM-semantic-segmentation](#nhm-semantic-segmentation)
2. [Deeplabv3-Plus-pytorch](#deeplabv3-plus-pytorch)
3. [YOLOv5-pytorch](#yolov5-pytorch)
4. [Paper-work](#paper-work)
5. [Reference](#reference)
6. [Contact](#contanct)

# NHM-semantic-segmentation

## Example output

![example-result](./example_result/nhm-semantic-segmentation/slides_rbgkslides_example_result.png)

# Deeplabv3-Plus-pytorch

## Quick start using

## Predicting images using existed model

1. choose the mode of prediction in [predict.py](./deeplabv3-plus-pytorch/predict.py), default is `mode = "dir_predict"`, here I use directory predict as example.
2. add images into `img` directory, either you can edit `dir_origin_path = "img/"` in [predict.py](./deeplabv3-plus-pytorch/predict.py) into anywhere you want.
3. choose the model you want, default is `"model_path": 'logs/best_epoch_weights.pth',`. You can edit it any model you want in [deeplab.py](./deeplabv3-plus-pytorch/deeplab.py)
4. run: `python predict.py`
5. the result will be in `./img_out` directory

> More details see [deeplabv3-plus-pytorch-README](./deeplabv3-plus-pytorch/README.md)

---

## How to train your own model

### If you only have origin images

1. put images into `./datasets/before` directory.
   <br>In Anaconda3, `pip install labelme=3.16.7`.
   <br>After installation, run `labelme`, and labelled you origin images and save \*.JSON with the same name as the images.
   <br>**IMPORTANT:** The origin image should be **24 bit depth RGB** image end with **`*.jpg`**
   <br>@see example here: [Example](./deeplabv3-plus-pytorch/datasets/before/)
2. `python labelme_to_deeplab.py`
   <br> The original JPG images are default to be saved in `"datasets/JPEGImages"`, and the labelled PNG images are default to be saved in `"datasets/SegmentationClass"`
   <br>**Note:** You should modified the `classes = ["_background_", "Barcode", "Label", "Color chart", "Scale"]` in [labelme_to_deeplab.py](./deeplabv3-plus-pytorch/labelme_to_deeplab.py) into your own number of labelled classes. You can't remove the `"\_background\_"`
3. Split the train/test/val datasets into deeplab's annotation. my default is `train/test/val = 80%/10%/10%`
   <br>`python get_annotation.py`
   <br>It will generate `ImageSets/Segmentation` in your dataset's root, with `test.txt`, `train.txt`, `trainval.txt`, `val.txt`
   <br>**NOTE:** If you want to modify the train/test/val you can modify the codes in [get_annotation.py](./deeplabv3-plus-pytorch/get_annotation.py)
4. `python train.py`

> More details see [deeplabv3-plus-pytorch-README](./deeplabv3-plus-pytorch/README.md)

### If you have both origin images and labelled images

1. Check the origin images are all **24 bit depth RGB** images end with **`*.jpg`**
   <br>Check the labelled images are all **8 bit depth RGB** images end with **`*.png`** with the same name as the origin images.
2. If your labelled PNG images can't meet the requirement, I provide [convert_lablled.py](./deeplabv3-plus-pytorch/convert_lablled.py) help you to either rename the images or change images from 24 bit depth into 8 bit depth.
3. If everything is okay, you can follow the steps above, get annotations and start training.

---

## How to evaluate your trained model

### If you already trained a model

1. modified the parameters in [get_miou.py](./deeplabv3-plus-pytorch/get_miou.py)
   <br> the parameters should be same as those before training.
2. `python get_miou.py`
3. The result will be in directory: `./miou_out`

> More details see [deeplabv3-plus-pytorch-README](./deeplabv3-plus-pytorch/README.md)

## Example output

![example-result](./example_result/deeplabv3-plus/RM0008192_example_result.jpg)

# YOLOv5-pytorch

## Example output

![example-result](./example_result/yolov5/RM0090530_example_result.jpg)

# Paper work

More detail about my project, you can see in my dissertation: [dissertation]()

# Reference

## NHM-semantic-segmentation

<ol>This project is modified from <li>origin repo: [semantic-segmentation](https://github.com/NaturalHistoryMuseum/semantic-segmentation)</li>
Fixed the bugs:

```
TypeError: Descriptors cannot not be created directly.

Solve:
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install protobuf==3.20.*
```

```
The system cannot find the file specified.
; No such file or directory

Solve:
set PYTHONPATH=${PYTHONPATH:`pwd`:`pwd`/slim}
```

```
counts = np.bincount(f['labels'][()].flatten(), minlength=len(self.class_to_idx))
MemoryError

Solve: resize the original images by using scripts: resize.py
```

```
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

Solve: add this line ahead runing code
torch.backends.cudnn.benchmark = True
```

```
ValueError: empty range for randrange() (0,-2, -2)
```

</ol>

---

## Deeplabv3-Plus-pytorch

<ol>This project is inspired by and modified from: 
<li>https://github.com/tensorflow/models/tree/master/research/deeplab</li>
<li>https://github.com/VainF/DeepLabV3Plus-Pytorch</li>
<li>https://github.com/bubbliiiing/deeplabv3-plus-pytorch</li>
</ol>

## YOLOv5-pytorch

# Contanct

Github: https://github.com/lyzsk

Email: sichu.huang@outlook.com

# LICENSE

[LICENSE](./LICENSE)
