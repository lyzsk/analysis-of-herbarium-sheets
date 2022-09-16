import os
from PIL import Image

# 去掉_label后缀
path = 'herbarium_sheets/mixseths/SegmentationClass'
for filename in os.listdir(path):
    os.rename(os.path.join(path, filename), os.path.join(path, filename[:-11] + '.png'))

# 24转8bit
for filename in os.listdir(path):
    img = Image.open(os.path.join(path, filename))
    print("convert %s into %s with 8bit...." % (img, filename))
    img = img.convert('P', palette=Image.ADAPTIVE, colors=8)
    img.save(os.path.join(path, filename))
