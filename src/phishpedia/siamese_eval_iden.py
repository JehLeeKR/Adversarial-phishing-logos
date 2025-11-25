from phishpedia.siamese import phishpedia_config
from phishpedia.siamese_pedia.inference import pred_siamese
from phishpedia.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from phishpedia.siamese_pedia.utils import brand_converter
from utils.utils import get_classes
import argparse
from PIL import Image
import pickle
import torch
import numpy as np
from collections import OrderedDict
import os
import torch.nn as nn

from phishpedia.siamese_pedia.utils import resolution_alignment


class QuantizeRelu(nn.Module):
    def __init__(self, step_size = 0.01):
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        mask = torch.ge(x, 0).bool() # mask for positive values
        quantize = torch.ones_like(x) * self.step_size
        out = torch.mul(torch.floor(torch.div(x, quantize)), self.step_size) # quantize by step_size
        out = torch.mul(out, mask) # zero-out negative values
        out = torch.abs(out) # remove sign
        return out

# write list to binary file
def write_list(data, file_path):
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list(file_path):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data

def get_max_sim(img):
    img_feat = pred_siamese(img, model)
    sim_list = logo_feat_list @ img_feat.T

    # get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    sim_list = np.array(sim_list)[idx]
    return sim_list[0]
        
    
def predict(img, threshold):
    img_feat = pred_siamese(img, model)
    # get cosine similarity with every protected logo
    sim_list = logo_feat_list @ img_feat.T
    pred_brand_list = file_name_list

    # get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_logolist = [Image.open(x) for x in pred_brand_list]
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_simlist = sim_list

    for j in range(3):
        # If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, 
        # otherwise might be false positive 
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        # If the largest similarity exceeds threshold
        if top3_simlist[j] >= threshold:
            predicted_brand = top3_brandlist[j]
            final_sim = top3_simlist[j]
            return predicted_brand, final_sim

        # Else if not exceed, try resolution alignment, see if can improve
        else:
            img, candidate_logo = resolution_alignment(img, top3_logolist[j])
            img_feat = pred_siamese(img, model)
            logo_feat = pred_siamese(candidate_logo, model)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= threshold:
                predicted_brand = top3_brandlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        # If there is a prediction, do aspect ratio check
        # if predicted_brand is not None:
        #     ratio_crop = img.size[0] / img.size[1]
        #     ratio_logo = top3_logolist[j].size[0] / top3_logolist[j].size[1]
        #     # aspect ratios of matched pair must not deviate by more than factor of 2.5
        #     if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
        #         continue  # did not pass aspect ratio check, try other
        #     # If pass aspect ratio check, report a match
        #     else:
        #         return predicted_brand, final_sim

    return None, top3_simlist[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate phishpedia")
    parser.add_argument("--weights_path", help="weights path", default='classification/phishpedia/src/siamese_pedia/finetune_bit.pth.tar')
    parser.add_argument("--targetlist_path", help="target list path", default='classification/datasets_logo_181/train/')
    parser.add_argument("--threshold", help="threshold value", type=float, default=0.83)
    parser.add_argument("--test_annotation_path", help="path for test images", type=str, default='classification/test_data.txt')
    parser.add_argument("--logo_feature_list", help="logo feature list for phishpedia", type=str, default='classification/phishpedia_data/step_relu/logo_feat_list_224.txt')
    parser.add_argument("--brand_list", help="brand list for phishpedia", type=str, default='classification/phishpedia_data/step_relu/brand_list_224.txt')
    parser.add_argument("--classes", help="class name", type=str, default='classification/datasets_logo_181/classes.txt')
    parser.add_argument("--tp_rates_path", help="true positive rate for phishpedia", type=str, default='classification/phishpedia_data/step_relu//iden_rates_224.txt')
    parser.add_argument("--fp_rates_path", help="false positive rate for phishpedia", type=str, default='classification/phishpedia_data/step_relu/fp_rates_224.txt')
    parser.add_argument("--use_step_relu", type=bool, default=True)

    args = parser.parse_args()
    test_annotation_path = args.test_annotation_path
    logo_feature_list_path = args.logo_feature_list
    brand_list_path = args.brand_list
    weights_path = args.weights_path
    classes_path = args.classes
    num_classes = 181
    starting_threshold = 0.817
    ending_threshold = 0.818
    stride = 0.002

    global model, logo_feat_list, file_name_list

    logo_feat_list = read_list(logo_feature_list_path)
    file_name_list = read_list(brand_list_path)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    if args.use_step_relu:
        for k, v in weights.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        # replace relu with defenselayer 
        model.body.block4.unit01.relu = QuantizeRelu()
        model.body.block4.unit02.relu = QuantizeRelu()
        model.body.block4.unit03.relu = QuantizeRelu()
    else:
        for k, v in weights.items():
            name = k.split('module.')[1]
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        
    model.to(device)
    model.eval()

    images = []
    labels = []
    class_names, _ = get_classes(classes_path)
    fp_test_annotation_path = 'classification/other_brand_logos/'
    fp_files = os.listdir(fp_test_annotation_path)
    with open(test_annotation_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            image_file_path = 'classification/' + line.strip().split(";")[1]
            label = int(line.split(';')[0])
            images.append(Image.open(image_file_path))
            labels.append(label)
            
    total = len(images)
    total_fp = len(fp_files)
    tp_rates = []

    test_threshold_count = {}
    threshold = starting_threshold
    while threshold <= ending_threshold:
        test_threshold_count[threshold] = 0
        threshold += stride

    while starting_threshold <= ending_threshold:
        tp = 0
        print('Current Threshold: %.2f' % (starting_threshold))
        # evaluate TP
        for idx, image in enumerate(images):
            brand, sim = predict(image, starting_threshold)
            if brand is not None:
                tp += 1
        tp_rate =  float(tp) / float(total)
        tp_rates.append(tp_rate)
        print('TP rate: %.5f' % tp_rate)
        starting_threshold += stride

if __name__ == '__main__':
    main()
