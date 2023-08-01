# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet-pytorch-main
File Name: move_files.py
Author: chenming
Create Date: 2022/3/11
Description：
-------------------------------------------------
"""
# import os
# import os.path as osp
# import shutil
# src_folder = "/scm/data/seg/data/kaggle_3m/train"
# folders = os.listdir(src_folder)
# for sub_folder in folders:
#     folder_path = osp.join(src_folder, sub_folder)
#     for file in os.listdir(folder_path):
#         print(file)
#         file_path = osp.join(folder_path, file)
#         if file.split(".")[0].split("_")[-1] == "mask":
#             shutil.copy(file_path, osp.join("/scm/data/seg/data/kaggle_3m/TrainLabels",  file.split("_mask")[0] + ".tif"))
#         else:
#             shutil.copy(file_path, osp.join("/scm/data/seg/data/kaggle_3m/TrainImages", file))


import os
import os.path as osp
import shutil
src_folder = "/scm/data/seg/data/kaggle_3m/test"
folders = os.listdir(src_folder)
for sub_folder in folders:
    folder_path = osp.join(src_folder, sub_folder)
    for file in os.listdir(folder_path):
        print(file)
        file_path = osp.join(folder_path, file)
        if file.split(".")[0].split("_")[-1] == "mask":
            shutil.copy(file_path, osp.join("/scm/data/seg/data/kaggle_3m/TestLabels",  file.split("_mask")[0] + ".tif"))
        else:
            shutil.copy(file_path, osp.join("/scm/data/seg/data/kaggle_3m/TestImages", file))

