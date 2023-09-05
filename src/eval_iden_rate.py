import os
from PIL import Image
import argparse
import pickle
import numpy as np
import torch

from classification import (Discriminator, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5, evaluate_by_class

import matplotlib.pyplot as plt

class Evaluator(Discriminator):
    def detect_image(self, image, clipping):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            preds = torch.softmax(torch.div(self.model(photo)[0], clipping), dim=-1).cpu().numpy()

        return preds
    
    def evaluate_tp(self, tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride):
        images = []
        labels = []
        tp_rate = []
        threshold = starting_threshold
            
        # evaluate TP
        with open(tp_test_annotation_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_file_path = 'classification/' + line.strip().split(";")[1]
                label = int(line.split(';')[0])
                images.append(Image.open(image_file_path))
                labels.append(label)
 
        threshold = starting_threshold
        while threshold <= ending_threshold:
            true_positive = 0
            print('Current threshold: ' + str("%.2f" % threshold))
        
            for idx, image in enumerate(images):
                preds = self.detect_image(image, clipping)
                probability = np.max(preds)

                if probability >= threshold:
                    true_positive += 1

            accuracy = (true_positive / len(images))
            print('Identification rate: ' + str("%.5f" % accuracy))
            tp_rate.append(accuracy)
            threshold += stride
            
        return tp_rate
            
        
        

parser = argparse.ArgumentParser(description='evaluate TP against FP')
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)
parser.add_argument('--starting_threshold', help='starting threshold', type=float, default=0.995)
parser.add_argument('--ending_threshold', help='ending threshold', type=float, default=0.998)
parser.add_argument('--stride', help='stride', type=float, default=0.002)
parser.add_argument('--tp_test_annotation_path', help='tp test annotation path', type=str, default='classification/test_data.txt')
parser.add_argument('--phishpedia_tp_rates', help='phishpedia tp', type=str, default='classification/phishpedia_data/step_relu/identification_rates_final.txt')

opt = parser.parse_args()
clipping = opt.output_clipping
starting_threshold = opt.starting_threshold
ending_threshold = opt.ending_threshold
stride = opt.stride
tp_test_annotation_path = opt.tp_test_annotation_path
phishpedia_tp_file = opt.phishpedia_tp_rates

# Read list to memory
def read_list(file_path):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data
    

# write list to binary file
def write_list(data, file_path):
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')


if __name__ == "__main__":
    # default backbone is vit_b_16
    discriminator = Evaluator()
#     tp_rate_vit = discriminator.evaluate_tp(tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride)
    
#     assert len(tp_rate_vit) == 20
#     write_list(tp_rate_vit, 'metrics_out/identification/vit.txt')
    
    # switch to swin_transformer_small
    # discriminator.switch_to_swin()
    tp_rate_swin = discriminator.evaluate_tp(tp_test_annotation_path, clipping, 0.975, 0.978, 0.002)
    # tp_rate_swin = discriminator.evaluate_tp(tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride)
    
#     assert len(tp_rate_swin) == 2
#     tp_rate_old = read_list('metrics_out/identification/swin.txt')
#     tp_rate_swin = tp_rate_old + tp_rate_swin
#     write_list(tp_rate_swin, 'metrics_out/identification/swin.txt')