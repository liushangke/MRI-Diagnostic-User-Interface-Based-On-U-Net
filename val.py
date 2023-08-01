import os

from PIL import Image
from tqdm import tqdm

from model_data.unet import Unet
from utils.utils_metrics import compute_mIoU, show_results


# 模型验证，三个参数，分别是测试集路径，结果保存路径和实际的标签路径
def val_main(test_dir="../ori_data/TestImages", result_dir="../ori_data/Results", gt_dir="../ori_data/TestLabels"):
    miou_mode = 0
    num_classes = 2  # 数据集的类别数目
    # name_classes = ["background", "target"] # 数据集的类名
    name_classes = ["background", "stroke"]  # 数据集的类名
    image_ids = [x.split(".")[0] for x in os.listdir(test_dir)]  # 数据集图片名称

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # print("Load model.")
        # 加载模型
        unet = Unet(model_path="logs/model_ori.pth") # todo 在这里修改模型的位置。
        print("模型加载完毕！")
        print("开始预测......")
        for image_id in tqdm(image_ids):
            # image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image_path = os.path.join(test_dir, image_id + ".bmp")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(result_dir, image_id + ".png"))
        print("预测完毕！")

    if miou_mode == 0 or miou_mode == 2:
        print("计算验证指标......")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, result_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("指标计算完毕，验证结果保存在val目录下")
        show_results("./val_results", hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == "__main__":
    val_main()
