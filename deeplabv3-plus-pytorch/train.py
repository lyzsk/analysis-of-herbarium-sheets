import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import show_config, download_weights
from utils.utils_fit import fit_one_epoch

'''
1.  input image is .jpg,
    there is no need to manually set the width and height of the input image, 
    when training it will auto resize the input image.
    
2.  The labelled image should be .png with 8 bit depth(Grayscale images),
    grayscale images will be auto converted to RGB images for training.

3.  loss value is used to judge whether the training model is going to converge. If it has decreasing trend by epoch,
    then it means that the val dataset's loss is decreasing, and the training is going well.
    If val dataset's loss is not going to change, it means the model is already converged.
    如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
    训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3.  The trained weights file is saved in the logs directory.
    Every Epoch(训练世代) has defined step(训练步长),
    every step will do a gradient descent
'''
if __name__ == "__main__":
    Cuda = True
    # distributed       用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    # @see https://github.com/bubbliiiing/deeplabv3-plus-pytorch/blob/main/train.py
    distributed = False
    # sync_bn           是否使用sync_bn，DDP模式多卡可用
    sync_bn = False
    # fp16              是否使用混合精度训练
    #                   可减少约一半的显存、需要pytorch1.7.1以上
    fp16 = False
    # num_classes       num of claesses + 1(_background_)
    num_classes = 5

    # mobilenet, xception
    backbone = "mobilenet"
    # pretrained        是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    pretrained = False

    # pretrained weights of the model are common to different datasets because the features are common
    # thbackbone's pre-train weight is used for feature extraction during training
    #
    # 训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    # if want to interrupts training during the training process, can set:
    # model_path = "logs/xxx.pt"
    # but i don't know if computer GPU over heat and locked down, should to use the last epoch's weight.
    # my expirence is there is no too big effect
    # To ensure the continuity of the model epoch, need to modify Freeze_Epoch and UnFreeze_Epoch
    #   
    # model_path = '' not using model weight.
    # if it is not used, the weights of the backbone part are too random,
    # the feature extraction effect is not obvious, and the results of network training will not be good
    #
    # 此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    # 如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    # 如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    model_path = "model_data/deeplab_mobilenetv2.pth"

    # downsample_factor     downsample factor 8 or 16
    #                       8 means 3 downsample count like theory but need more memory of GPU
    #                       I use 4 to solve the OOM
    downsample_factor = 16

    # size of input image
    input_shape = [512, 512]

    # train has two steps: freeze, unfreeze
    # freeze step to meet the training needs of insufficient machine performance
    # freeze need smaller memory of GPU
    # set Freeze_Epoch = UnFreeze_Epoch to only freeze training for me
    #      
    # 1. start training with pretrained weights for the entire model:
    #   Adam：
    #       Freeze:
    #       Init_Epoch = 0，
    #       Freeze_Epoch = 50，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = True，
    #       optimizer_type = 'adam'，
    #       Init_lr = 5e-4，
    #       weight_decay = 0
    #
    #       No freeze:
    #       Init_Epoch = 0，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = False，
    #       optimizer_type = 'adam'，
    #       Init_lr = 5e-4，
    #       weight_decay = 0
    #   SGD：
    #       Freeze:
    #       Init_Epoch = 0，
    #       Freeze_Epoch = 50，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = True，
    #       optimizer_type = 'sgd'，
    #       Init_lr = 7e-3，
    #       weight_decay = 1e-4
    #
    #       No freeze:
    #       Init_Epoch = 0，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = False，
    #       optimizer_type = 'sgd'，
    #       Init_lr = 7e-3，
    #       weight_decay = 1e-4
    #
    #       UnFreeze_Epoch can be in range 100-300
    #
    # 2. start training from the pretrained weights of the backbone network:
    #   Adam：
    #       Freeze:
    #       Init_Epoch = 0，
    #       Freeze_Epoch = 50，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = True，
    #       optimizer_type = 'adam'，
    #       Init_lr = 5e-4，weight_decay = 0
    #
    #       No freeze:
    #       Init_Epoch = 0，
    #       UnFreeze_Epoch = 100，
    #       Freeze_Train = False，
    #       optimizer_type = 'adam'，
    #       Init_lr = 5e-4，weight_decay = 0
    #
    #   SGD：
    #       Freeze:
    #       Init_Epoch = 0，
    #       Freeze_Epoch = 50，
    #       UnFreeze_Epoch = 120，
    #       Freeze_Train = True，
    #       optimizer_type = 'sgd'，
    #       Init_lr = 7e-3，
    #       weight_decay = 1e-4
    #
    #       No freeze:
    #       Init_Epoch = 0，
    #       UnFreeze_Epoch = 120，
    #       Freeze_Train = False，
    #       optimizer_type = 'sgd'，
    #       Init_lr = 7e-3，
    #       weight_decay = 1e-4
    #
    # Since the training starts from the pre-trained weights of the backbone network,
    # the weights of the backbone are not necessarily suitable for semantic segmentation,
    # and more training is required to jump out of the local optimal solution
    # UnFreeze_Epoch can be in range 120-300
    # Adam相较于SGD收敛的快一些 因此UnFreeze_Epoch理论上可以小一点 但依然推荐更多的Epoch
    #
    # batch_size：受到BatchNorm层影响，batch_size最小为2，不能为1。
    # normally, Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size
    # too much difference between freeze batch and unfreeze batch size will affect automatic adjustment of learning rate
    #
    # 冻结阶段训 模型的主干被冻结了 特征提取网络不发生改变
    # Init_Epoch            模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    # Freeze_Epoch          模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    # Freeze_batch_size     模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    # 解冻阶段 模型的主干不被冻结了 特征提取网络会发生改变
    # UnFreeze_Epoch            模型总共训练的epoch
    # Unfreeze_batch_size       模型在解冻后的batch_size
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # Freeze_Train      是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练
    Freeze_Train = True

    # Init_lr           maximum learning rate, sgd default learning rate is 0.01
    #                   for Adam optimizer:  Init_lr=5e-4
    #                   for SGD optimizer:   Init_lr=7e-3
    # Min_lr            min learning rate is as sgd's 0.01
    Init_lr = 7e-3
    Min_lr = Init_lr * 0.01

    # optimizer_type    adam, sgd
    # momentum          优化器内部使用到的momentum参数
    # weight_decay      weight decay to prevent overfitting
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4  # using adam will weight_decay bug, weight_decay = 0 when using adm
    # lr_decay_type    'step', 'cos'
    lr_decay_type = 'cos'
    # save log every 5 epoch
    save_period = 5
    save_dir = 'logs'
    # eval_flag         evaluate during training, evaluate the val dataset
    # eval_period       evaluation takes a lot of time, and frequent evaluation will lead to very slow training
    # this eval mAp is val dataset's mAp hence will be different with get_miou.py output
    eval_flag = True
    eval_period = 5

    # Herbarium_path  数据集路径
    Herbarium_path = 'herbarium_sheets'
    # num_classes < 10: True
    # num_classes > 10，batch_size > 10: True
    # num_classes > 10，batch_size < 10: False
    # Dice Loss is equivalent to investigating from a global perspective. 
    # BCE zooms in pixel by pixel from a microscopic perspective, with complementary angles
    # TODO: I don't know if I need to enable the dice loss
    #
    # @see https://github.com/LIVIAETS/boundary-loss
    # @see https://blog.csdn.net/longshaonihaoa/article/details/111824916
    # @see deeplabv3_training.py
    dice_loss = True
    # enable/disable focal loss to prevent the imbalance of positive and negative samples
    focal_loss = False

    # every class's loss weight，default to be balanced
    cls_weights = np.ones([num_classes], np.float32)
    # num_workers       multithread reading data
    #                   1 close multithread (recommend for bad GPU heat dissipation capacity and small memory)
    #                   original is 4 sad
    num_workers = 1

    # GPU setting
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # download weights
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    # record loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # torch 1.2 has no amp, torch > 1.7.1 can use fp16
    # if fp16:
    #     from torch.cuda.amp import GradScaler as GradScaler
    #
    #     scaler = GradScaler()
    # else:
    #     scaler = None
    scaler = None
    model_train = model.train()

    # 多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # 多卡平行运行
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # read txt of train, test, val
    # IMPORTANT! NEED TO MODIFY HERE!
    with open(os.path.join(Herbarium_path, "wyoming_combine_origin/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(Herbarium_path, "wyoming_combine_origin/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            num_train=num_train, num_val=num_val
        )

        # total_step: count of all gradient descents
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset too small, need more data...')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] %s optimizer need to set total_step > %d \033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] num of train: %d，Unfreeze_batch_size: %d，total epoch: %d个Epoch，total step: "
                "%d。\033[0m" % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Because total step is: %d，please set < %d，set epoch: %d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

    # 主干特征提取网络特征通用，冻结训练可以加快训练速度
    # 也可以在训练初期防止权值被破坏。
    # Init_Epoch为起始世代
    # Interval_Epoch为冻结训练的世代
    # Epoch总训练世代
    # OOM need to reduce Batch_size
    if True:
        UnFreeze_flag = False
        # 冻结一定部分训练
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 判断当前batch_size，自适应调整学习率
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # set optimizer
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # get the formula for the learning rate drop
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, Herbarium_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, Herbarium_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

        # record eval map curve
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, Herbarium_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # start training model
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 如果模型有冻结学习部分,则解冻,并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # 判断当前batch_size，自适应调整学习率
                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # 获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
