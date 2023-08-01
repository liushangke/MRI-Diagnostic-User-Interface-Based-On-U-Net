# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet-pytorch-main
File Name: final_train.py
Author: chenming
Create Date: 2022/3/7
Description：
-------------------------------------------------
"""
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch_no_val
import os.path as osp


# 模型训练主流程
# 数据集路径，训练轮数，数据集批次大小（显存小于4g的请设置为1）, 学习
def train_main(dataset_path='../ori_data/', epochs=50, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 本机使用的设备，如果没有GPU将会自动调整为CPU 无需调整
    num_classes = 2  # 要进行分类的类别，医学影像数据集一般设置为2
    backbone = "resnet50"  # 主干特征提取网络，有两种选择vgg和resnet50
    pretrained = True  # 是否采用预训练的模型,设置为true的时候将会自动下载主干特征网络在预训练网络上的模型。
    model_path = ""  # 是否加载中断模型
    input_shape = [512, 512]  # 图片输入大小
    dice_loss = False  # 医学一箱数据集设置为True可以一定精度，如果种类较多尽量不要使用
    focal_loss = False  # 设置为True可解决正负样本不均衡的问题
    cls_weights = np.ones([num_classes], np.float32)
    Freeze_Train = True  # 是否冻结主干特征提取网络，可以提高模型训练速度。
    num_workers = 0  # window下设置为0即可
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model_train = model.train()
    # model_train = torch.nn.DataParallel(model)
    # cudnn.benchmark = True
    model_train = model_train.to(device)
    loss_history = LossHistory("logs/", val_loss_flag=False)  # 模型保存路径
    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    # with open(os.path.join(dataset_path, "train.txt"), "r") as f:
    #     train_lines = f.readlines()
    train_lines = [x.split(".")[0] for x in os.listdir(osp.join(dataset_path, "TrainImages"))]

    epoch_step = len(train_lines) // batch_size

    if epoch_step == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    optimizer = optim.Adam(model_train.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, dataset_path)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)

    if Freeze_Train:
        model.freeze_backbone()

    for epoch in range(epochs):
        fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, epochs, device,
                             dice_loss, focal_loss, cls_weights, num_classes)
        lr_scheduler.step()


if __name__ == '__main__':
    train_main()  # 启动训练过程，模型将会保存在logs目录下
