**简体中文 | [English](./README.md)**

[![license](https://img.shields.io/github/license/lyzsk/analysis-of-herbarium-sheets.svg?style=plastic&logo=github)](https://github.com/lyzsk/analysis-of-herbarium-sheets/deeplabv3-plus-pytorch/blob/master/LICENSE) ![python3](https://img.shields.io/badge/language-Python3-14274E?style=plastic&logo=python) ![anaconda3](https://img.shields.io/badge/environment-Anaconda3-394867?style=plastic&logo=anaconda) ![pytorch](https://img.shields.io/badge/environment-Pytorch-394867?style=plastic&logo=pytorch)

---

# Get start

-   [installation](#installation)
-   [train with default dataset](#train-with-default-dataset)
-   [predict with default dataset](#predict-with-default-dataset)
-   [train with custom dataset](#train-with-custom-dataset)
-   [predict with custom dataset](#predict-with-custom-dataset)

# Installation

```
<!-- Manual installation -->
conda create -n deeplabv3plus python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

pip install scipy==1.9.1

pip install matplotlib==3.5.3

pip install opencv-python==4.6.0.66

pip install tqdm==4.64.1

pip install h5py==3.7.0

pip install tensorboard==2.10.0

<!-- Auto installation -->
conda create -n deeplabv3plus python=3.9

pip install -r requirements.txt
```

> :warning: IMPORTANT: if you want to use GPU mode, check you download related CUDA and CUDNN.

# Train with default dataset

If you want to train with default dataset (herbarium_sheets.zip):

1. Download [herbarium_sheets.zip](https://zenodo.org/record/4448186)

    Unzip the `herbarium_sheets.zip`

    move the \*.JPG images (unlabelled images) to `deeplabv3-plus-pytorch/herbarium_sheets/wyoming_combine_origin/JPEGImages`

    move the \*\_\_labels.png images (labelled images) to `deeplabv3-plus-pytorch/herbarium_sheets/wyoming_combine_origin/SegmentationClass` and run `convert_labelled.py`:

    ```python
    python convert_labelled.py
    ```

2. run `get_annotation.py` to generate test.txt, train.txt, trainval.txt, val.txt in `deeplabv3-plus-pytorch/herbarium_sheets/wyoming_combine_origin/ImagesSets/Segmentation`:

    ```python
    python get_annotation.py
    ```

3. run `train.py`:

    ```python
    python train.py
    ```

# Predict with default dataset

If you want to predict with default dataset (Wyoming dataset):

1. Download the zip files below:

    - [Balsamorhiza_incana](https://rmh.uwyo.edu/images/cardiff/Balsamorhiza_incana.zip)
    - [Balsamorhiza_sagittata](https://rmh.uwyo.edu/images/cardiff/Balsamorhiza_sagittata.zip)
    - [Geum_aleppicum](https://rmh.uwyo.edu/images/cardiff/Geum_aleppicum.zip)
    - [Geum_macrophyllum_var_perincisum](https://rmh.uwyo.edu/images/cardiff/Geum_macrophyllum_var_perincisum.zip)
    - [Mertensia_lanceolata](https://rmh.uwyo.edu/images/cardiff/Mertensia_lanceolata.zip)
    - [Mertensia_viridis](https://rmh.uwyo.edu/images/cardiff/Mertensia_viridis.zip)
    - [Oxytropis_besseyi_var_besseyi](https://rmh.uwyo.edu/images/cardiff/Oxytropis_besseyi_var_besseyi.zip)
    - [Oxytropis_lambertii_var_lambertii](https://rmh.uwyo.edu/images/cardiff/Oxytropis_lambertii_var_lambertii.zip)
    - [Packera_fendleri](https://rmh.uwyo.edu/images/cardiff/Packera_fendleri.zip)
    - [Packera_multilobata](https://rmh.uwyo.edu/images/cardiff/Packera_multilobata.zip)

2. move the desired image you want to predict into `deeplabv3-plus-pytorch/img` directory

3. run `predict.py`:

    ```python
    python predict.py
    ```

# Train with custom dataset

1. Create pixel-level labels using Labelme

2. modify and run `labelme_to_deeplab.py`:

    ```python
    # 1. change the root dirs and classes into yours
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"
    classes = ["_background_", "Barcode", "Label", "Color chart", "Scale"]

    # 2. run
    python labelme_to_deeplab.py
    ```

3. modify and run `get_annotation.py`:

    ```python
    # 1. change the dirs
    segfilepath = os.path.join(Herbarium_path, 'wyoming_combine_origin/SegmentationClass')

    saveBasePath = os.path.join(Herbarium_path, 'wyoming_combine_origin/ImageSets/Segmentation')

    # 2. run
    python get_annotation.py
    ```

4. modify `dataloader.py`:

    ```python
    # 1. change dirs name
    jpg = Image.open(
        os.path.join(os.path.join(self.dataset_path, "wyoming_combine_origin/JPEGImages"), name + ".jpg"))

    png = Image.open(
        os.path.join(os.path.join(self.dataset_path, "wyoming_combine_origin/SegmentationClass"), name + ".png"))
    ```

5. modify and run `train.py`:

    ```python
    # 1. change num_classes to the same number of your labelme classes
    num_classes = 5

    # 2. run
    python train.py
    ```

# Predict with custom dataset

1. modify `deeplab.py`:

    ```python
    # 1. change num_classes to the same number of your training
    "num_classes": 5,
    ```

2. modify and run `predict.py`

    ```python
    # 1. change name_classes to the same as you used in training
    name_classes = ["background", "Barcode", "Label", "Color chart", "Scale"]

    # 2. put your image into `img` directory

    # 3. run
    python predict.py
    ```
