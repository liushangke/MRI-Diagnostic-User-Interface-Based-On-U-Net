#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ai_train_code 
@File    ：gen_txt.py
@Author  ：ChenmingSong
@Date    ：2021/12/31 18:42 
@Description：
'''
import os
import random
import numpy as np

# todo 改为你标注文件的路径即可
# annotations_foder_path = "E:/biye/gogogo/english/detection_yolov5/official/data/origin/VOC2012/Annotations"
annotations_foder_path = "F:/bbbbbbbbbb/tmp/seg/eye/Training_Images"
names = os.listdir(annotations_foder_path)
real_names = [name.split(".")[0] for name in names]
print(real_names)
random.shuffle(real_names)
print(real_names)
length = len(real_names)
split_point = int(length * 0.2)

val_names = real_names[:split_point]
train_names = real_names[split_point:]

# 开始生成文件
np.savetxt('val.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('test.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('train.txt', np.array(train_names), fmt="%s", delimiter="\n")
print("txt文件生成完毕，请放在VOC2012的ImageSets/Main的目录下")

# np.savetxt('bbbbb.txt', np.array(real_names), fmt="%s", delimiter="\n")
