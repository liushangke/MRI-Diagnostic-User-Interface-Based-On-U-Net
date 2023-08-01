# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from model_data.unet import Unet


def predict(model_path, img_path):
    unet = Unet(model_path=model_path)
    image = Image.open(img_path)
    r_image = unet.detect_image(image)
    r_image.show()


if __name__ == "__main__":
    predict(model_path="logs/brain_model.pth", img_path="images/TCGA_CS_4941_19960909_11.tif")
