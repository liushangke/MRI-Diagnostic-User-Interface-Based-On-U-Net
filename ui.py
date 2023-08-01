# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: ui.py.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model_data.unet import Unet
import numpy as np
import uuid
import os
# 窗口主类
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('基于Unet的脑肿瘤图像分割')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        # # 初始化视频读取线程
        self.origin_shape = ()
        # 加载网络，图片单通道，分类为1。
        unet = Unet(model_path="logs/model_ori.pth")
        self.is_det = False
        self.is_download = False
        self.model = unet
        self.initUI()
        self.folder = "tmp"

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        download_img_button = QPushButton("下载结果")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        download_img_button.clicked.connect(self.download)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        download_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        download_img_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(download_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        real_about_layout = QVBoxLayout()

        grid_widget = QWidget()
        about_layout = QGridLayout()
        name = QLabel("患者姓名")
        age = QLabel("年龄")
        time = QLabel("就诊时间")
        suggest = QLabel("诊断意见")
        self.name_edit = QLineEdit()
        self.age_edit = QLineEdit()
        self.time_edit = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.suggest_edit = QTextEdit()

        about_layout.setSpacing(10)
        about_layout.addWidget(name, 1, 0)
        about_layout.addWidget(self.name_edit, 1, 1)

        about_layout.addWidget(age, 2, 0)
        about_layout.addWidget(self.age_edit, 2, 1)

        about_layout.addWidget(time, 3, 0)
        about_layout.addWidget(self.time_edit, 3, 1)

        about_layout.addWidget(suggest, 4, 0)
        about_layout.addWidget(self.suggest_edit, 4, 1)

        go_button = QPushButton("提交")
        go_button.setFont(font_main)
        go_button.setStyleSheet("QPushButton{color:white}"
                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                "QPushButton{background-color:rgb(48,124,208)}"
                                "QPushButton{border:2px}"
                                "QPushButton{border-radius:5px}"
                                "QPushButton{padding:5px 5px}"
                                "QPushButton{margin:5px 5px}")
        go_button.clicked.connect(self.go)

        about_title = QLabel('诊断信息填写')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        # about_img = QLabel()
        # about_img.setPixmap(QPixmap('images/UI/qq.png'))
        # about_img.setAlignment(Qt.AlignCenter)
        # # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        # label_super = QLabel()  # todo 更换作者信息
        # label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        # label_super.setFont(QFont('楷体', 16))
        # label_super.setOpenExternalLinks(True)
        # # label_super.setOpenExternalLinks(True)
        # label_super.setAlignment(Qt.AlignRight)
        # about_layout.addWidget(about_title)
        # about_layout.addStretch()
        # about_layout.addWidget(about_img)
        # about_layout.addStretch()
        # about_layout.addWidget(label_super)
        grid_widget.setFont(font_main)
        grid_widget.setLayout(about_layout)
        real_about_layout.addWidget(about_title)
        real_about_layout.addWidget(grid_widget)
        real_about_layout.addWidget(go_button)
        about_widget.setLayout(real_about_layout)
        # about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(about_widget, '报告填写')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg *.bmp')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
            self.is_det = False
            self.is_download = False

    '''
    ***检测图片***
    '''

    def detect_img(self):
        # model = self.model
        # output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        image = Image.open(source)
        result_image = self.model.detect_image_ui(image)
        result_image.save("images/tmp_result.jpg")
        pred = cv2.imread("images/tmp_result.jpg")
        im0 = cv2.resize(pred, self.origin_shape)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.is_det = True
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    def download(self):
        if self.is_det == False:
            QMessageBox.warning(self, "无法下载", "请先进行检测")
        elif self.is_download:
            QMessageBox.information(self, "重复下载", "请勿重复下载结果")
        else:
            src_img_path = "images/tmp/single_result.jpg"
            uuid_folder_name = str(uuid.uuid4())
            os.mkdir("record/" + uuid_folder_name)
            self.folder = uuid_folder_name
            shutil.copy2(src_img_path, "record/" + self.folder)
            QMessageBox.information(self, "下载完成", "图片已下载到{}".format(uuid_folder_name))
            self.is_download = True

    def go(self):
        # 获取信息
        name = str(self.name_edit.text())
        age = str(self.age_edit.text())
        time = str(self.time_edit.text())
        suggest = str(self.suggest_edit.toPlainText())
        if name == "":
            QMessageBox.warning(self, "不能为空", "请填写患者姓名")
        if age == "":
            QMessageBox.warning(self, "不能为空", "请填写患者年龄")
        if time == "":
            QMessageBox.warning(self, "不能为空", "请填写就诊时间")
        if suggest == "":
            QMessageBox.warning(self, "不能为空", "请填写诊断意见")
        if self.folder == "tmp":
            QMessageBox.warning(self, "无法生成", "请先下载结果图片")
        else:
            with open("record/{}/reslt.txt".format(self.folder), "w", encoding="utf-8") as f:
                f.writelines(["姓名：{}\n".format(name), "年龄：{}\n".format(age), "就诊时间：{}\n".format(time),
                              "诊断意见：{}\n".format(suggest)])
                QMessageBox.information(self, "报告已生成", "报告已下载到{}".format(self.folder))

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
