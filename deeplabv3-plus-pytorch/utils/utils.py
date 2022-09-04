import numpy as np
import torch
from PIL import Image


# convert the image to an RGB image to prevent an error in the prediction of grayscale images
# only support prediction for RGB images, others will change into RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# resize input image
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


# get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    download_urls = {
        'mobilenet': 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth',
        'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
