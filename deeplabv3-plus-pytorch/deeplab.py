import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# IMPORTANT! NEED TO MODIFY:    model_path, backbone, num_classes
# 如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
class DeeplabV3(object):
    _defaults = {
        # For myself: model_path指向logs文件夹下的权值文件
        # lower loss in val dataset does not mean higher miou，
        # it only means that the weights generalize better on the validation set
        # "model_path"        : 'model_data/deeplab_mobilenetv2.pth',
        # IMPORTANT! NEED TO MODIFY HERE!
        "model_path": 'logs/best_epoch_weights.pth',

        # num of classes + 1(background)
        "num_classes": 5,

        # banckbone： mobilenet, xception
        "backbone": "mobilenet",

        # 输入图片的大小
        "input_shape": [512, 512],

        # downsamle rate: 8 or 16, should be same as training
        "downsample_factor": 16,

        # mix_type参数用于控制检测结果的可视化方式
        # mix_type = 0的时候代表原图与生成的图进行混合
        # mix_type = 1的时候代表仅保留生成的图
        # mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        "mix_type": 0,

        # 是否使用Cuda
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # red, green, yellow, blue, purple, light blue, grey, dark red, light red, light green...
        # (original VOC has 21classes so 21 colors)
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # get model
        self.generate()

        show_config(**self._defaults)

    # get all classes
    def generate(self, onnx=False):
        # 载入模型与权值
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone,
                           downsample_factor=self.downsample_factor, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        # convert into RGB png, because:
        #   1. only png has the classes of each pixel.
        #   2. we use RGB color the assign labels.
        # @see utils.py
        image = cvtColor(image)
        # copy image, make a backup
        old_img = copy.deepcopy(image)
        # get width, height
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # add gray bars to the image to achieve undistorted resize (letterbox image)
        # 也可以直接resize进行识别
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # normalization + 添加上batch_size维度, put the channel at the first dimension
        # This is the Pytorch requirement
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # transfer the image_data into the Pytorch format
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # put images into nets to predict
            pr = self.net(images)[0]

            # use torch.nn.functional.softmax, get the class of each pixel
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # crop the gray bars added before
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # 进行图片的resize
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # take out the class of each pixel
            pr = pr.argmax(axis=-1)

        # 计数
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            # assign class label with color into the objects
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            # 将新图片转换成Image的形式, resize from the 512,512 into original size
            image = Image.fromarray(np.uint8(seg_img))

            # 将新图与原图及进行混合
            # 如果想要原图和分割图不混合，可以把blend参数设置成False, see in predict.py
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # 将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            # 将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        # convert the image to an RGB image to prevent an error in the prediction of grayscale images
        # only support prediction for RGB images, others will change into RGB
        image = cvtColor(image)
        # add gray bars to the image to achieve undistorted resize
        # 也可以直接resize进行识别
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # 需要pytorch1.7.1以上
    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        # convert the image to an RGB image to prevent an error in the prediction of grayscale images
        # only support prediction for RGB images, others will change into RGB
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # add gray bars to the image to achieve undistorted resize
        # 也可以直接resize进行识别
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 图片传入网络进行预测
            pr = self.net(images)[0]
            # 取出每一个像素点的种类
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # 将灰条部分截取掉
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # 进行图片的resize
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # 取出每一个像素点的种类
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
