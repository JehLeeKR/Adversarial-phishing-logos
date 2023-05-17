import os
from PIL import Image

import numpy as np
import torch

from classification import (Discriminator, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5, evaluate_by_class

import matplotlib.pyplot as plt

# ------------------------------------------------------#
#   test_annotation_path    测试图片路径和标签
# ------------------------------------------------------#
test_annotation_path = '../data/others/'

class Evaluator(Discriminator):
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对图片进行不失真的resize
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # ---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds


threshold = []
FP_rate = []

if __name__ == "__main__":
    discriminator = Evaluator()
    files = os.listdir(test_annotation_path)
    starting_thresh = 0.997

    while starting_thresh < 0.999:
        above_threshold = 0
        threshold.append(starting_thresh)
        print(starting_thresh)
        
        for file in files:
            image_path = test_annotation_path + file
            image = Image.open(image_path)
            
            preds = discriminator.detect_image(image)
            probability = np.max(preds)

            if probability > starting_thresh :
                above_threshold += 1

        print(above_threshold)
        print(len(files))
        fp = (above_threshold / len(files))
        print(fp)
        FP_rate.append(fp)
        starting_thresh += 0.001
    

    plt.plot(threshold, FP_rate)
    plt.xlabel('Threshold')
    plt.ylabel('FP rate')
    plt.title('FP against threshold plot')
    plt.savefig('../result/plot/FP_threshold_swin.png')
    plt.show()
