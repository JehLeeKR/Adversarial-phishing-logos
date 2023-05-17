import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import os
from PIL import Image

from .nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                                        preprocess_input, show_config, get_backbone_name)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
# --------------------------------------------#
class Discriminator(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": '../classification/exp_full_vit_181_sgd/best_epoch_weights.pth',
        "swin_model_path": '../classification/exp_full_swin_181_sgd/best_epoch_weights.pth',
        "classes_path": '../classification/datasets_logo_181/classes.txt',
        # --------------------------------------------------------------------#
        #   输入的图片大小
        # --------------------------------------------------------------------#
        "input_shape": [224, 224],
        # --------------------------------------------------------------------#
        #   所用模型种类：
        #   resnet50
        #   vit_b_16
        #   swin_transformer_small
        # --------------------------------------------------------------------#
        "backbone": 'vit_b_16',
        # --------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        # --------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化classification
    # ---------------------------------------------------#
    def __init__(self, discriminator_type='vit', **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   获得种类
        # ---------------------------------------------------#
        self.model = None
        # self.backbone = get_backbone_name(discriminator_type)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # model_path = 'model_path'
        # if self.get_defaults(model_path) == '':
        #     self.model_path = get_pretrained_weights_path(self.backbone)
        # else:
        #     self.model_path = self.get_defaults(model_path)
        self.init_weights()
        if discriminator_type == 'swin':
            self.switch_to_swin()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def init_weights(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
                                 'swin_transformer_base']:
            self.model = get_model_from_name[self.backbone](num_classes=self.num_classes, pretrained=False)
        else:
            self.model = get_model_from_name[self.backbone](input_shape=self.input_shape, num_classes=self.num_classes,
                                                            pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def switch_to_swin(self):
        self.backbone = 'swin_transformer_small'
        self.model_path = self.swin_model_path
        self.init_weights()
    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
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
            photo = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        # ---------------------------------------------------#
        #   获得所属种类
        # ---------------------------------------------------#
        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        # ---------------------------------------------------#
        #   绘图并写字
        # ---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' % (class_name, probability))
        plt.show()
        return class_name
    
    def detect_image_with_clipping(self, image, clipping):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(torch.div(self.model(photo)[0], clipping), dim=-1).cpu().numpy()

        return preds
    
    def evaluate_fp(self, clipping, thresholds, fp_test_annotation_path):
        fp_rate = []
        fp_rate_count = {}
        fp_files = os.listdir(fp_test_annotation_path)
        for i in thresholds:
            fp_rate_count[i] = 0
            
        total = len(fp_files)
        for file in fp_files:
            if file.startswith('.'):
                total -= 1
                continue
            image_path = fp_test_annotation_path + file
            image = Image.open(image_path)
            
            preds = self.detect_image_with_clipping(image, clipping)
            probability = np.max(preds)
            for thres in thresholds:
                if probability >= thres:
                    fp_rate_count[thres] += 1

        
        for i in thresholds:
            fp = float(fp_rate_count[i]) / float(total)
            fp_rate.append(fp)
            print('FP rate: ' + str("%.5f" % fp))
            
        return fp_rate
