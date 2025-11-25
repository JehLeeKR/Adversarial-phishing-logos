import os
from PIL import Image
import argparse
import pickle
import numpy as np
import torch

from classification import (Discriminator, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from classification.utils.utils_metrics import evaluteTop1_5, evaluate_by_class

import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm

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
   

    def evaluate_pred(self, str_positive_file_list,  str_negative_dir, clipping):
        
        # Load Positive Samples
        image_positive = []
        label_positive = []

        # Image Path list
        with open(str_positive_file_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_file_path = 'classification/' + line.strip().split(";")[1]
                label = int(line.split(';')[0])
                image_positive.append(Image.open(image_file_path))
                label_positive.append(label)

        # Load Negative Samples
        image_negative = []
        label_negative = []
        # Image Files
        for str_filename in os.listdir(str_negative_dir):
            str_filepath = os.path.join(str_negative_dir, str_filename)        

            if os.path.isdir(str_filepath):
                continue
            
            image_negative.append(Image.open(str_filepath))
                

        # Predict
        list_pred = []
        for image in tqdm(image_positive):
            dict_pred = {}
            preds = self.detect_image(image, clipping)
            probability = np.max(preds)
            dict_pred['Label'] = 1
            dict_pred['Pred'] = probability
            list_pred.append(dict_pred)

        for image in tqdm(image_negative):
            dict_pred = {}
            preds = self.detect_image(image, clipping)
            probability = np.max(preds)
            dict_pred['Label'] = 0
            dict_pred['Pred'] = probability
            list_pred.append(dict_pred)

        df_pred = pd.DataFrame(list_pred)
        return df_pred
    

    def evaluate_tp_fp(self, fp_test_annotation_path, tp_test_annotation_path,
                        clipping, starting_threshold, 
                        ending_threshold, stride):
        images = []
        labels = []
        fp_rate = []
        tp_rate = []
        fp_files = os.listdir(fp_test_annotation_path)
        threshold = starting_threshold
        
        # evaluate FP
        while threshold <= ending_threshold:
            above_threshold = 0
            print('Current threshold: ' + str("%.3f" % threshold))
        
            for file in fp_files:
                if file.startswith('.'):
                    continue
                image_path = fp_test_annotation_path + file
                image = Image.open(image_path)
            
                preds = self.detect_image(image, clipping)
                probability = np.max(preds)

                if probability >= threshold :
                    above_threshold += 1

            fp = (above_threshold / len(fp_files))
            fp_rate.append(fp)
            print('FP rate: ' + str("%.5f" % fp))
            threshold += stride
            
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
            print('Current threshold: ' + str("%.3f" % threshold))
        
            for idx, image in enumerate(images):
                preds = self.detect_image(image, clipping)
                probability = np.max(preds)

                # if probability >= threshold and np.argmax(preds) == labels[idx]:
                #     true_positive += 1
                if probability >= threshold:
                    true_positive += 1

            accuracy = (true_positive / len(images))
            print('TP rate: ' + str("%.5f" % accuracy))
            tp_rate.append(accuracy)
            threshold += stride
            
        return fp_rate, tp_rate
                
    def evaluate_fp(self, fp_test_annotation_path, tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride):
        fp_rate = []
        test_threshold_count = {}
        threshold = starting_threshold
        while threshold <= ending_threshold:
            test_threshold_count[threshold] = 0
            threshold += stride
            
        fp_files = os.listdir(fp_test_annotation_path)
        total = len(fp_files)
        for file in fp_files:
            threshold = starting_threshold
            if file.startswith('.'):
                continue
            image_path = fp_test_annotation_path + file
            image = Image.open(image_path)
            
            preds = self.detect_image(image, clipping)
            probability = np.max(preds)
            while threshold <= ending_threshold and probability >= threshold:
                test_threshold_count[threshold] += 1
                threshold += stride
        
        threshold = starting_threshold
        while threshold <= ending_threshold:
            fp = (test_threshold_count[threshold] / len(fp_files))
            fp_rate.append(fp)
            print('FP rate: ' + str("%.5f" % fp))
            threshold += stride
        
        return fp_rate
        

parser = argparse.ArgumentParser(description='evaluate TP against FP')
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)
parser.add_argument('--starting_threshold', help='starting threshold', type=float, default=0.61)
parser.add_argument('--ending_threshold', help='ending threshold', type=float, default=1.0)
parser.add_argument('--stride', help='stride', type=float, default=0.02)
parser.add_argument('--fp_test_annotation_path', help='fp test annotation path', type=str, default='classification/other_brand_logos/')
parser.add_argument('--tp_test_annotation_path', help='tp test annotation path', type=str, default='classification/test_data.txt')
parser.add_argument('--phishpedia_tp_rates', help='phishpedia tp', type=str, default='classification/phishpedia_data/step_relu/tp_rates_224.txt')
parser.add_argument('--phishpedia_fp_rates', help='phishpedia fp', type=str, default='classification/phishpedia_data/step_relu/fp_rates_224.txt')

opt = parser.parse_args()
clipping = opt.output_clipping
starting_threshold = opt.starting_threshold
ending_threshold = opt.ending_threshold
stride = opt.stride
fp_test_annotation_path = opt.fp_test_annotation_path
tp_test_annotation_path = opt.tp_test_annotation_path
phishpedia_tp_file = opt.phishpedia_tp_rates
phishpedia_fp_file = opt.phishpedia_fp_rates

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
    # tp_rate_siamese = read_list(phishpedia_tp_file)[4:14]
    # fp_rate_siamese = read_list(phishpedia_fp_file)[4:14]
#     tp_rate_siamese_plus = read_list(phishpedia_tp_file)
#     tp_rate_siamese = read_list('classification/phishpedia_data/tp_rates_224.txt')
#     fp_rate_siamese = read_list(phishpedia_fp_file)
#     tp_rate_swin = read_list('metrics_out/tp_fp/tp_rates_swin.txt')
#     # print(tp_rate_swin)
#     assert len(tp_rate_siamese) == 20
#     assert len(fp_rate_siamese) == 20
    
#     with open('classification/train_data.txt', "r") as f:
#         lines = f.readlines()
#         print(len(lines))
    
    # ViT
    # default backbone is vit_b_16
    discriminator = Evaluator(discriminator_type = 'vit')
    pytorch_total_params = sum(p.numel() for p in discriminator.model.parameters())
    #print(pytorch_total_params)
    # discriminator.switch_to_swin()
    # df_pred_ViT = discriminator.evaluate_pred(tp_test_annotation_path, fp_test_annotation_path, 2.5)
    # df_pred_ViT.to_csv('./plots/csv/ViT_Pred.csv')

    # Swin
    # discriminator.switch_to_swin()
    # pytorch_total_params = sum(p.numel() for p in discriminator.model.parameters())
    # df_pred_Swin = discriminator.evaluate_pred(tp_test_annotation_path, fp_test_annotation_path, 2.5)
    # df_pred_Swin.to_csv('./plots/csv/Swin_Pred.csv')
    # exit()
    # fp_rate_vit, tp = discriminator.evaluate_tp_fp(fp_test_annotation_path, tp_test_annotation_path, 2.5, 0.98, 0.980001, 0.00005)
    # print(fp_rate_vit)
    # print(tp)
    
#     tp_rate_vit_latter = read_list('metrics_out/tp_fp/tp_rates_vit.txt')
#     fp_rate_vit_latter = read_list('metrics_out/tp_fp/fp_rates_vit.txt')
    
#     fp_rate_vit += fp_rate_vit_latter
#     tp_rate_vit += tp_rate_vit_latter
    
    # assert len(fp_rate_vit) == 20
#     assert len(tp_rate_vit) == 20
    
    # write_list(fp_rate_vit, 'metrics_out/tp_fp/fp_rates_vit.txt')
    # write_list(tp_rate_vit, 'metrics_out/tp_fp/tp_rates_vit.txt')
    
    # switch to swin_transformer_small
    discriminator.switch_to_swin()
    pytorch_total_params = sum(p.numel() for p in discriminator.model.parameters())
    print(pytorch_total_params)
    # fp_rate_swin = discriminator.evaluate_fp(fp_test_annotation_path, tp_test_annotation_path, 2.5, starting_threshold, ending_threshold, stride)
    # print(fp_rate_swin)
    # print(tp_rate_swin)
    
#     tp_rate_swin_orig = read_list('metrics_out/tp_fp/tp_rates_swin.txt')
#     fp_rate_swin_orig = read_list('metrics_out/tp_fp/fp_rates_swin.txt')
#     print("original TP:")
#     print(tp_rate_swin_orig)
#     print("original FP:")
#     print(fp_rate_swin_orig)
#     print("clipping 2:")
#     print(fp_rate_swin)
#     print(tp_rate_swin)

    
#     fp_rate_swin, tp_rate_swin = discriminator.evaluate_tp_fp(fp_test_annotation_path, 
#                                       tp_test_annotation_path, 2.7, 
#                                       starting_threshold, ending_threshold, 
#                                       stride)
#     print("clipping 2.7:")
#     print(fp_rate_swin)
#     print(tp_rate_swin)
    
#     fp_rate_swin, tp_rate_swin = discriminator.evaluate_tp_fp(fp_test_annotation_path, tp_test_annotation_path, 2.3, starting_threshold, ending_threshold, stride)
#     print("clipping 2.3:")
#     print(fp_rate_swin)
#     print(tp_rate_swin)
    # fp_rate_swin = discriminator.evaluate_fp(fp_test_annotation_path, tp_test_annotation_path, 2.5, 0.970, 0.971, 0.0001)
          
          
#     fp_rate_swin += fp_rate_swin_latter
#     tp_rate_swin += tp_rate_swin_latter
    
    # assert len(fp_rate_swin) == 20
#     assert len(tp_rate_swin) == 20
    # write_list(fp_rate_swin, 'metrics_out/tp_fp/fp_rates_swin.txt')
#     write_list(tp_rate_swin, 'metrics_out/tp_fp/tp_rates_swin.txt')
    
    
#     plt.plot(fp_rate_vit, tp_rate_vit, color='blue', linewidth = 1.5,
#          marker='D', markerfacecolor='blue', markersize=4, label='ViT')
#     plt.plot(fp_rate_swin, tp_rate_swin, color='green', linewidth = 1.5, linestyle = 'dashed',
#          marker='s', markerfacecolor='green', markersize=4, label='Swin')
#     plt.plot(fp_rate_siamese, tp_rate_siamese, color='red', linewidth = 1.5, linestyle = 'dashdot',
#          marker='v', markerfacecolor='red', markersize=4, label='Siamese++')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive rate')
#     plt.xscale('log')
#     plt.legend(loc='lower right')
#     plt.savefig('metrics_out/roc/roc_clipping_' + str(clipping) + '_20' + '.png')
#     plt.show()
