**简体中文 | [English](./README.md)**

[![license](https://img.shields.io/github/license/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/blob/master/LICENSE) [![forks](https://img.shields.io/github/forks/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/members) [![stars](https://img.shields.io/github/stars/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/stargazers) ![python3](https://img.shields.io/badge/language-Python3-14274E?style=plastic&logo=python) ![anaconda3](https://img.shields.io/badge/environment-Anaconda3-394867?style=plastic&logo=anaconda) ![pytorch](https://img.shields.io/badge/framework-Pytorch-394867?style=plastic&logo=pytorch)

---

# About The Project

这个项目是用 监督学习语义分割 (Deeplabv3+, U-Net, PSPNet) 和 半监督学习语义分割 (NaturalHistoryMeuseum(NHM) semantic segmentation) 来训练 herbarium sheets 和 microscopy slides (植物标本和显微镜标本?). 然后用 监督学习目标检测 (YOLOv5) 来识别植物标本对象, 从而检测 classification 的异常

> **_如果你喜欢这个项目或者这个项目对你有帮助, 别忘了点赞哦_** :star:

训练集: NMWHS, MNHNS, MIXSETHS.

-   下载链接: [Download Link](https://zenodo.org/record/4448186)
-   为了扩展训练集，我还用 [labelme](https://anaconda.org/conda-forge/labelme) 手动新增了 200 张 Wyoming University 的 dataset.

预测数据 are provided by Wyoming University.

-   下载链接:
    -   [Balsamorhiza_incana](https://rmh.uwyo.edu/images/cardiff/Balsamorhiza_incana.zip)
    -   [Balsamorhiza_sagittata](https://rmh.uwyo.edu/images/cardiff/Balsamorhiza_sagittata.zip)
    -   [Geum_aleppicum](https://rmh.uwyo.edu/images/cardiff/Geum_aleppicum.zip)
    -   [Geum_macrophyllum_var_perincisum](https://rmh.uwyo.edu/images/cardiff/Geum_macrophyllum_var_perincisum.zip)
    -   [Mertensia_lanceolata](https://rmh.uwyo.edu/images/cardiff/Mertensia_lanceolata.zip)
    -   [Mertensia_viridis](https://rmh.uwyo.edu/images/cardiff/Mertensia_viridis.zip)
    -   [Oxytropis_besseyi_var_besseyi](https://rmh.uwyo.edu/images/cardiff/Oxytropis_besseyi_var_besseyi.zip)
    -   [Oxytropis_lambertii_var_lambertii](https://rmh.uwyo.edu/images/cardiff/Oxytropis_lambertii_var_lambertii.zip)
    -   [Packera_fendleri](https://rmh.uwyo.edu/images/cardiff/Packera_fendleri.zip)
    -   [Packera_multilobata](https://rmh.uwyo.edu/images/cardiff/Packera_multilobata.zip)

# Table of content:

1. [NHM-semantic-segmentation](#nhm-semantic-segmentation)
2. [Deeplabv3-Plus-pytorch](#deeplabv3-plus-pytorch)
    - [installation](#installation)
    - [get start](#get-start)
    - [example result](#example-result)
3. [U-Net-Pytorch](#u-net-pytorch)
    - [installation](#installation-1)
    - [get start](#get-start-1)
    - [example result](#example-result-1)
4. [PSPNet-Pytorch](#pspnet-pytorch)
    - [installation](#installation-2)
    - [get start](#get-start-2)
    - [example result](#example-result-2)
5. [YOLOv5-pytorch](#yolov5-pytorch)
    - [installation](#installation-3)
    - [get start](#get-start-3)
    - [example result](#example-result-3)

# NHM-semantic-segmentation

Details @see https://github.com/NaturalHistoryMuseum/semantic-segmentation

因为源项目根本跑不了所以我重写了一部分，实在看不懂。。 :cold_sweat:, what I did:

1. 在启动代码第一行添加 `torch.backends.cudnn.benchmark = True` :neutral_face:
2. 重写了 `trainmodel.py` and `predict.py`, 至少现在这个版本可以训练 microscope slides. 但是还是不能训练 herbarium sheets, 我也不想管了，其他人的代码抄起来太爽了 :neutral_face:

## Example result

![nhm-example](./example_result/nhm-semantic-segmentation/slides_rbgkslides_example_result.png)

# Deeplabv3-Plus-pytorch

## Installation

```
<!-- 手动安装 -->
conda create -n deeplabv3plus python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

pip install scipy==1.9.1

pip install matplotlib==3.5.3

pip install opencv-python==4.6.0.66

pip install tqdm==4.64.1

pip install h5py==3.7.0

pip install tensorboard==2.10.0

<!-- 自动安装 -->
conda create -n deeplabv3plus python=3.9

pip install -r requirements.txt
```

> :warning: IMPORTANT: 如果你要用 GPU 训练,请检查安装自己机器对应的 CUDA 和 CUDNN

## Get start

```python
# 训练
python train.py

# 预测
python predict.py

# 输出 mIOU.png, mPA.png, mPrecision.png, mRecall.png, confusion_matrix.csv, detection-results
python get_miou.py

# labelme JSON 转 deeplab PNG
python labelme_to_deeplab.py

# 把 24-bit depth PNG 转成 8-bit depth PNG
python convert_labelled.py

# 生成 train-test-val TXT
python get_annotation.py
```

> :rainbow: 详情见： [README](./deeplabv3-plus-pytorch/README.md) or [README.CN](./deeplabv3-plus-pytorch/README.CN.md)

## Example result

![deeplabv3plus-example](./example_result/deeplabv3-plus/RM0008156_example_result.jpg)

# U-Net-Pytorch

## Installation

```
<!-- Manual installation -->
conda create -n unet python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

pip install scipy==1.9.1

pip install matplotlib==3.5.3

pip install opencv-python==4.6.0.66

pip install tqdm==4.64.1

pip install h5py==3.7.0

pip install tensorboard==2.10.0

pip install onnx==1.12.0

<!-- Auto installation -->
conda create -n unet python=3.9

pip install -r requirements.txt
```

## Get start

```python
# train
python train.py

# predict
python predict.py

# get mIOU.png, mPA.png, mPrecision.png, mRecall.png, confusion_matrix.csv, detection-results
python get_miou.py

# labelme JSON to unet PNG
python labelme_to_unet.py

# convert 24-bit depth PNG to 8-bit depth PNG
python convert_labelled.py

# generate train-test-val TXT
python get_annotation.py
```

> :rainbow: More details see [README](./unet-pytorch/README.md) or [README.CN](./unet-pytorch/README.CN.md)

## Example result

![unet-example](./example_result/unet/RM0008156_example_result.jpg)

# PSPNet-Pytorch

## Installation

```
<!-- Manual installation -->
conda create -n pspnet python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

pip install scipy==1.9.1

pip install matplotlib==3.5.3

pip install opencv-python==4.6.0.66

pip install tqdm==4.64.1

pip install h5py==3.7.0

pip install tensorboard==2.10.0

pip install onnx==1.12.0

<!-- Auto installation -->
conda create -n pspnet python=3.9

pip install -r requirements.txt
```

## Get start

```python
# train
python train.py

# predict
python predict.py

# get mIOU.png, mPA.png, mPrecision.png, mRecall.png, confusion_matrix.csv, detection-results
python get_miou.py

# labelme JSON to pspnet PNG
python labelme_to_pspnet.py

# convert 24-bit depth PNG to 8-bit depth PNG
python convert_labelled.py

# generate train-test-val TXT
python get_annotation.py
```

> :rainbow: More details see [README](./pspnet-pytorch/README.md) or [README.CN](./pspnet-pytorch/README.CN.md)

## Example result

![pspnet-example](./example_result/pspnet/RM0008180_example_result.jpg)

# YOLOv5-pytorch

> :warning: IMPORTANT: This project is still in TODO mode<br>
> :warning: IMPORTANT: This project is still in TODO mode<br>
> :warning: IMPORTANT: This project is still in TODO mode<br>
> 重要的事情说三遍 Important things said three times

## Installation

```
<!-- Manual installation -->
conda create -n yolov5 python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

pip install scipy==1.9.1

pip install matplotlib==3.5.3

pip install opencv-python==4.6.0.66

pip install tqdm==4.64.1

pip install h5py==3.7.0

pip install tensorboard==2.10.0

pip install onnx==1.12.0

<!-- Auto installation -->
conda create -n yolov5 python=3.9

pip install -r requirements.txt
```

## Get start

```python
# train
python train.py

# predict
python predict.py

# get detection-results, ground-truth, images-optional, results
python get_map.py

# generate train-test-val TXT
python voc_annotation.py
```

> :rainbow: More details see [README](./yolov5-pytorch/README.md) or [README.CN](./yolov5-pytorch/README.CN.md)

## Example result

![yolov5-example-1](./example_result/yolov5/RM0090530_example_result.jpg)

![yolov5-example-2](./example_result/yolov5/RM0008156_example_result.png)

# Reference

https://github.com/NaturalHistoryMuseum/semantic-segmentation

https://github.com/ultralytics/yolov5

https://github.com/tensorflow/models

https://github.com/ggyyzm/pytorch_segmentation

https://github.com/bonlime/keras-deeplab-v3-plus

https://github.com/matterport/Mask_RCNN

https://github.com/leekunhee/Mask_RCNN

https://github.com/anylots/detection

https://github.com/yyccR/yolov5_in_tf2_keras

https://github.com/wkentaro/labelme/

https://github.com/xiaotudui/PyTorch-Tutorial

https://github.com/bubbliiiing/deeplabv3-plus-pytorch

https://github.com/bubbliiiing/pspnet-pytorch

https://github.com/bubbliiiing/unet-pytorch

> 跪了，感谢大佬们上传代码给我抄，要不然真的交不了作业了 :tired_face::tired_face::tired_face:

# LICENSE

[LICENSE](./LICENSE)

---

> TODO: built with Flask output API into React + Springboot + SpringMVC + MybatisPlus + Redis + RabbitMQ + MongoDB + MySQL + Elasticsearch + Druid + FastDFS + Kubernetes + Docker+++++++++++++++++++++++ 淦 救救孩子吧
