from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from material.models.generators import ResnetGenerator, weights_init
#from data import get_training_set, get_test_set
from utils.dataloader import DataGenerator
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
from classification import Discriminator
from customized_loss import CrossEntropyLossWithThreshold
from gap_util import custom_pil_loader
import pickle

plt.switch_backend('agg')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='evaluate fooling ratio against FP rate')
parser.add_argument('--dataVal', type=str, default='~/autodl-tmp/gap/classification/datasets_logo_181/test',
                    help='data val root')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--expname', type=str, default='metrics_out/fooling_ratio', help='experiment name, output folder')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIterTest', type=int, default=500, help='Iterations in each Epoch')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--checkpoint_customize', type=str, default='vit_mag_10_customized_clipping/netG_model_epoch_188_foolrat_92.9568099975586.pth', help='path to starting checkpoint')
parser.add_argument('--checkpoint_baseline', type=str, default='vit_mag_10_baseline_clipping/netG_model_epoch_183_foolrat_85.95297029702971.pth', help='path to starting checkpoint')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--starting_threshold', help='threshold used by the discriminator for classification', type=float, default=0.61)
parser.add_argument('--ending_threshold', help='maximum threshold used by the discriminator for classification', type=float, default=1.00)
parser.add_argument('--stride', help='threshold stride', type=float, default=0.02)
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)

opt = parser.parse_args()

if not torch.cuda.is_available():
    raise Exception("No GPU found.")

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIterTest = opt.MaxIterTest
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)
clipping = opt.output_clipping
starting_threshold = opt.starting_threshold
ending_threshold = opt.ending_threshold
stride = opt.stride

test_threshold = []
test_threshold_count = {}
tp_count = {}
fooling_ratio_base = []
fooling_ratio_customized = []
threshold = starting_threshold
while threshold <= ending_threshold:
    test_threshold.append(threshold)
    test_threshold_count[threshold] = 0
    threshold += stride

# define normalization means and stddevs
center_crop = 224
input_shape = [224, 224]

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

print('===> Loading datasets')

test_annotation_path = './classification/test_data.txt'
path_prefix = './classification'
fp_test_annotation_path = 'classification/other_brand_logos/'

with open(test_annotation_path, encoding='utf-8') as f:
    test_lines = f.readlines()
    
test_set = DataGenerator(test_lines, input_shape, False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

discriminator = Discriminator()
pretrained_discriminator = discriminator.model.cuda(gpulist[0])
pretrained_discriminator.eval()
pretrained_discriminator.volatile = True

# magnitude
mag_in = opt.mag_in

# will use model paralellism if more than one gpu specified
netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

                
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

                    
def load_generator(is_baseline):
    if is_baseline:
        print("=> loading checkpoint '{}'".format(opt.checkpoint_baseline))
        netG.load_state_dict(torch.load(opt.checkpoint_baseline, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(opt.checkpoint_baseline))
    else:
        print("=> loading checkpoint '{}'".format(opt.checkpoint_customize))
        netG.load_state_dict(torch.load(opt.checkpoint_customize, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(opt.checkpoint_customize))
        
                    
def test_fooling_ratio(is_base):
    for i in test_threshold_count:
        test_threshold_count[i] = 0
        tp_count[i] = 0
    load_generator(is_base)       
    netG.eval()
    total = 0

    for itr, (image, class_label) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, 'test')

        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        outputs_recon = pretrained_discriminator(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_discriminator(image.cuda(gpulist[0]))
        
        outputs_recon = torch.softmax(torch.div(outputs_recon, clipping), dim=-1)
        outputs_orig = torch.softmax(torch.div(outputs_orig, clipping), dim=-1)
        
        recon_val, _ = torch.max(outputs_recon, 1)
        orig_val, _ = torch.max(outputs_orig, 1)
        total += image.size(0)

        recon_val = recon_val.tolist()
        orig_val = orig_val.tolist()
        for idx, val in enumerate(orig_val):
            for _, threshold_val in enumerate(test_threshold):
                if val >= threshold_val and recon_val[idx] < threshold_val:
                    test_threshold_count[threshold_val] = test_threshold_count[threshold_val] + 1
                    
                if val >= threshold_val:
                    tp_count[threshold_val] = tp_count[threshold_val] + 1

    for _, threshold_val in enumerate(test_threshold):
        if is_base:
            fooling_ratio_base.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
        else:
            fooling_ratio_customized.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))

def test_fp():
    fp_rate_base = discriminator.evaluate_fp(clipping, test_threshold, fp_test_annotation_path)
    return fp_rate_base

                    
def plot_fooling_ratio_against_fp(fp_rate_base):
    for idx, threshold in enumerate(test_threshold):
        print('Threshold: ' + str("%.2f" % threshold) + ', False Positive Rate: ' + str("%.5f" % fp_rate_base[idx]) + ', Baseline Fooling Ratio: ' + str("%.2f" % fooling_ratio_base[idx]) + ', Fooling Ratio of Customized Generator: ' + str("%.2f" % fooling_ratio_customized[idx]))
    plt.plot(fp_rate_base, fooling_ratio_base, color='blue', linewidth = 1.5, linestyle = 'dashed',
         marker='D', markerfacecolor='blue', markersize=4, label='Generator with Untargeted Training')
    plt.plot(fp_rate_base, fooling_ratio_customized, color='green', linewidth = 1.5,
         marker='s', markerfacecolor='green', markersize=4, label='Generator with Targeted Training')
    # plt.scatter(fp_rate_base, fooling_ratio_base, color='blue', marker='D', label='Generator with Untargeted Training')
    # plt.scatter(fp_rate_base, fooling_ratio_customized, color='green', marker='s', label='Generator with Targeted Training')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Fooling Ratio')
    plt.xscale('log')
    plt.xlim(right=1.1)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('metrics_out/fooling_ratio_vit/fooling_ratios_comparison.png')
    plt.show()

def plot_fooling_ratio_against_threshold():
    plt.plot(test_threshold, fooling_ratio_base, color='blue', linewidth = 1.5, linestyle = 'dashed',
         marker='D', markerfacecolor='blue', markersize=4, label='Generator with Untargeted Training')
    plt.plot(test_threshold, fooling_ratio_customized, color='green', linewidth = 1.5,
         marker='s', markerfacecolor='green', markersize=4, label='Generator with Targeted Training')
    plt.xlabel('Threshold Value')
    plt.ylabel('Fooling Ratio')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('metrics_out/fooling_ratio_vit/fooling_ratios_against_threshold_comparison.png')
    plt.show()
                    
def normalize_and_scale(delta_im, mode='train'):
    delta_im = delta_im + 1  # now 0..2
    delta_im = delta_im * 0.5  # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:, c, :, :] = (delta_im[:, c, :, :].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.testBatchSize
    for i in range(len(delta_im)):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
            mag_in_scaled_c = mag_in / (255.0 * stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

if __name__ == "__main__":
    # print('Testing fooling ratio with baseline generator...')
    # test_fooling_ratio(is_base=True)
#     fooling_ratio_base = [0.7926772388059702, 0.7988331388564761, 0.8042056074766355, 0.8092120645312134, 0.81437265917603, 0.8202247191011236, 0.8253336455162725, 0.8319268635724332, 0.8385734396996715, 0.847672778561354, 0.8509185115402732, 0.8585930122757318, 0.8671245855045002, 0.8760704091341579, 0.883169739047163, 0.8918853840597158, 0.9050509956289461, 0.9193627450980392, 0.9339717741935484, 0.9547910150696617]
    fooling_ratio_base = read_list('metrics_out/fooling_ratio_vit_old/vit_fooling_ratio_base_final.txt')
    fooling_ratio_customized = read_list('metrics_out/fooling_ratio_vit_old/vit_fooling_ratio_final.txt')
    fp_rate_vit = read_list('metrics_out/tp_fp/fp_rates_vit.txt')
    # print('Testing fooling ratio with customized generator...')
    # test_fooling_ratio(is_base=False)
    # print('Testing FP rate...')
    # fp_rate_base = test_fp()
    print('Plotting...')
    plot_fooling_ratio_against_fp(fp_rate_vit)
    plt.clf()
    plot_fooling_ratio_against_threshold()