**[简体中文](./README.CN.md) | English**

[![license](https://img.shields.io/github/license/lyzsk/segmentation.svg?style=plastic&logo=github)](https://github.com/lyzsk/segmentation/blob/master/LICENSE) ![python3](https://img.shields.io/badge/language-Python3-14274E?style=plastic&logo=python) ![anaconda3](https://img.shields.io/badge/environment-Anaconda3-394867?style=plastic&logo=anaconda) ![pytorch](https://img.shields.io/badge/environment-Pytorch-394867?style=plastic&logo=pytorch)

---

# Before use

1. All the codes runs with Python3.6 in anaconda environment.
2. The `SegmentationClass` folder in datasets should only contains images with `.png` format with 8 bit depth!<br>
   The `JPEGImages` folder in datasets should only contains images with '.jpg' format with 24 bit depth.
   If you have problem with transorming 24 bit depth -> 8 bit, I provided `convert_lablled.py` file, you can run it after editing the `path` variable.
    ```
    python convert_lablled.py
    ```
3. Don't make `num_workers > 1` if your GPU has poor heat dissipation or memory < 8GB.
