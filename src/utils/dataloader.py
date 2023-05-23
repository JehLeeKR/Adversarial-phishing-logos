import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import sys

# str_utils_abs_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(str_utils_abs_path, '..'))

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
from phishpedia.siamese_eval_util import *

import torch.utils.data as data
from PIL import Image, ImageOps
import os


class DataGeneratorV2(data.Dataset):
    def __init__(self, annotation_lines, data_root, transform=None, is_test=False):
        self.data_root = data_root
        self.transform = transform
        self.annotation_lines = annotation_lines
        self.is_test = is_test   
        self.preprocess_labels  = {}
        self.preprocess_sim = {}
        if is_test:
            self.file_path = 'siamese_mag_10_new/preprocess_sim_test.txt'
            with open(self.file_path, 'rb') as fp:
                self.preprocess_sim = pickle.load(fp)
                print('Done loading similarity list')
        else:
            self.file_path = 'siamese_mag_10_new/preprocess_labels_train.txt'
            with open(self.file_path, 'rb') as fp:
                self.preprocess_labels = pickle.load(fp)
                print('Done loading label list')
        
        # with open(self.file_path, 'rb') as fp:
        #     self.preprocess_labels = pickle.load(fp)
        #     print('Done loading label list')
        # print(len(self.preprocess_labels))
        
        # if self.is_test:
        #     with open('siamese_mag_10/preprocess_sim_test.txt', 'rb') as fp:
        #         self.preprocess_sim = pickle.load(fp)
        #         print('Done loading similarity list')
#         for i in range(len(self.annotation_lines)):
#             print("Processing: " + str(i))
#             annotation_path = self.annotation_lines[i].split(';')[1].split()[0]
#             img_path_full = os.path.join(self.data_root, annotation_path)
#             image = Image.open(img_path_full)
#             img = cvtColor(image)
#             img = ImageOps.expand(img, (
#                 (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
#                 (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

#             if self.transform is not None:
#                 img = self.transform(img)
                       
#             label_feat, sim = get_max_sim(img)
#             self.preprocess_sim[annotation_path] = sim
#             self.preprocess_labels[annotation_path] = label_feat
        
#         if is_test:
#             with open('siamese_mag_10_new/preprocess_sim_test.txt', 'wb') as fp:
#                 pickle.dump(self.preprocess_sim, fp)
#                 print('Done writing list into a binary file')
#         else:
#             with open('siamese_mag_10_new/preprocess_labels_train.txt', 'wb') as fp:
#                 pickle.dump(self.preprocess_labels, fp)
#                 print('Done writing list into a binary file')
        print("Data Processing Done...")

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        img_path_full = os.path.join(self.data_root, annotation_path)
        image = Image.open(img_path_full)
        img   = cvtColor(image)
        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        if self.transform is not None:
            img = self.transform(img)

        if self.is_test:
            return img, self.preprocess_sim[annotation_path]
        else:
            return img, self.preprocess_labels[annotation_path]


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True, autoaugment_flag=True, transform=None, prefix=None, is_siamese=False):
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.random             = random
        self.autoaugment_flag   = autoaugment_flag
        self.transform          = transform
        self.prefix             = prefix
        self.is_siamese = is_siamese
            
        
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        if self.prefix is not None:
            annotation_path = os.path.join(self.prefix, annotation_path)
        image = Image.open(annotation_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        img   = cvtColor(image)
        if self.is_siamese:
            img = ImageOps.expand(img, (
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), 
                fill=(255, 255, 255))

        if self.transform is not None:
            img = self.transform(img)
        # if self.autoaugment_flag:
        #     image = self.AutoAugment(image, random=self.random)
        # else:
        #     image = self.get_random_data(image, self.input_shape, random=self.random)
        else:
            img = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(';')[0])
        return img, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
            
def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
    return images, targets
