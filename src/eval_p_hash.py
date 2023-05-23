from __future__ import print_function
import argparse
import os
import math
from tqdm import tqdm
import sys

from math import log10
import matplotlib.pyplot as plt
import numpy as np
import imagehash

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# # Tools Path
# str_src_abs_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(str_src_abs_path, '.'))

from PIL import Image
from collections import defaultdict

from material.models.generators import ResnetGenerator, weights_init
from utils.dataloader import DataGenerator
from gap_util import str_project_root_path, g_gpu_available
from gap_util import gz_pickle_load, gz_pickle_dump
from gap_util import custom_pil_loader
from gap_util import get_training_set, get_test_set

import pandas as pd


# define normalization means and stddevs
model_dimension = 256
center_crop = 224
input_shape = [224, 224]

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                std=stddev_arr)

data_transform = transforms.Compose([
    # transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])



def create_pertubated_images(str_data_tag, opt):
    #str_data_tag = 'train' # test

    
    print('===> Loading datasets')
    #test_annotation_path = f"{str_project_root_path}/data/test_data.txt"
    test_annotation_path = f"{str_project_root_path}/data/{str_data_tag}_data.txt"
    path_prefix = f"{str_project_root_path}/data"

    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
        
    test_set = DataGenerator(test_lines, input_shape, False, 
                             autoaugment_flag=False, transform=data_transform,
                             prefix=path_prefix)
    
    testing_data_loader = DataLoader(dataset=test_set, shuffle=False, 
                                     batch_size=opt.testBatchSize,num_workers=opt.threads)

    print('===> Building model')

    # will use model paralellism if more than one gpu specified
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', 
                           act_type='relu', gpu_ids=gpulist)

    # resume from checkpoint if specified
    if opt.checkpoint:
        str_checkpoint_path = os.path.join(str_project_root_path, 'model', opt.checkpoint)
        if os.path.isfile(str_checkpoint_path):
            print("=> loading checkpoint '{}'".format(opt.checkpoint))
            netG.load_state_dict(torch.load(str_checkpoint_path,
                                            map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{}'".format(opt.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(opt.checkpoint))
            netG.apply(weights_init)
    else:
        netG.apply(weights_init)


    netG.eval()        
    list_label = []
    #with open(f"{str_project_root_path}/data/test_data.txt") as f_read:
    with open(f"{str_project_root_path}/data/{str_data_tag}_data.txt") as f_read:
        for str_line in f_read:
            str_label = str_line.split(';')[1].split('/')[-2].lower()
            list_label.append(str_label)
        
    int_file_index = 0
    for (image, _) in tqdm(testing_data_loader):
        
        if g_gpu_available:
            image = image.cuda(gpulist[0])
            delta_im = netG(image)
            delta_im = normalize_and_scale(delta_im, opt, str_data_tag)
            recons = torch.add(image.cuda(gpulist[0]), 
                            delta_im[0:image.size(0)].cuda(gpulist[0]))
        else:
            # CPU Mode            
            delta_im = netG(image)
            delta_im = normalize_and_scale(delta_im, opt, str_data_tag)
            recons = torch.add(image, delta_im[0:image.size(0)])

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        delta_im_temp = torch.zeros(delta_im.size())
        image_scale = image
        for c2 in range(3):
            recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            image_scale[:, c2, :, :] = (image_scale[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]

        
        # Save Scaled and Perturbated.
        for int_img_idx in range(len(image)):                    
            str_label = list_label[int_file_index]
            int_subcnt = 0
            
            str_img_basename = f"Logo_{str_label}_{int_subcnt}.png"
            str_org_img_path = f"{str_project_root_path}/data/protected/org/{str_label}/{str_img_basename}"
            while os.path.exists(str_org_img_path):
                int_subcnt += 1
                str_img_basename = f"Logo_{str_label}_{int_subcnt}.png"
                str_org_img_path = f"{str_project_root_path}/data/protected/org/{str_label}/{str_img_basename}"

            str_org_img_path = f"{str_project_root_path}/data/protected/org/{str_label}/{str_img_basename}"
            str_scaled_org_img_path = f"{str_project_root_path}/data/protected/scaled_org/{str_label}/{str_img_basename}"
            str_pertubated_img_path = f"{str_project_root_path}/data/protected/pertubated/{str_label}/{str_img_basename}"
        
            if not os.path.exists(f"{str_project_root_path}/data/protected/org/{str_label}"):
                os.makedirs(f"{str_project_root_path}/data/protected/org/{str_label}")

            if not os.path.exists(f"{str_project_root_path}/data/protected/scaled_org/{str_label}"):
                os.makedirs(f"{str_project_root_path}/data/protected/scaled_org/{str_label}")

            if not os.path.exists(f"{str_project_root_path}/data/protected/pertubated/{str_label}"):
                os.makedirs(f"{str_project_root_path}/data/protected/pertubated/{str_label}")
                
            torchvision.utils.save_image(image[int_img_idx], str_org_img_path)
            torchvision.utils.save_image(image_scale[int_img_idx], str_scaled_org_img_path)
            torchvision.utils.save_image(recons[int_img_idx], str_pertubated_img_path)
            int_file_index += 1
    return

import glob

def calculate_p_hash() -> list[(str, float)]:    
    dist_data = []
    str_org_img_path = f"{str_project_root_path}/data/protected/org/Logo_*.png"
    list_img = glob.glob(str_org_img_path)
    
    # Save Scaled and Perturbated.
    for str_img_path in tqdm(list_img):
        str_img_basename = os.path.basename(str_img_path)
        str_org_img_path = f"{str_project_root_path}/data/protected/org/{str_img_basename}"
        str_scaled_org_img_path = f"{str_project_root_path}/data/protected/scaled_org/{str_img_basename}"
        str_pertubated_img_path = f"{str_project_root_path}/data/protected/pertubated/{str_img_basename}"
                
        original_hash = imagehash.phash(Image.open(str_scaled_org_img_path))
        reconstructed_hash = imagehash.phash(Image.open(str_pertubated_img_path))

        hash_dist = original_hash - reconstructed_hash
        dist_data.append((str_scaled_org_img_path, hash_dist))        
    return dist_data
        
def normalize_and_scale(delta_im, opt, mode='train'): 
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
            mag_in_scaled_c = opt.mag_in / (255.0 * stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im


def plot_distribution_graph(freq_count):
    lists = sorted(freq_count.items())# sorted by key, return a list of tuples
    hash_dist, freq = zip(*lists)
    freq = list(freq)# unpack a list of pairs into two tuples
    for i in range(len(freq)):
        freq[i] = freq[i] * 100
    plt.plot(hash_dist, tuple(freq), color='red', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='red', markersize=3)
    for a,b in zip(hash_dist, tuple(freq)): 
        plt.text(a, b + 0.05, '%.2f%%' % (b), fontsize=6)
    #plt.title('Hamming Distance between Original Image and Perturbed Image')
    plt.ylabel('Percentage')
    plt.xlabel('Hamming Distance')
    plt.savefig(f"{str_project_root_path}/result/plot/hamming_dist/protected/hamming_dist_distribution.pdf", format='pdf')
    print("Saved plots.")

def plot_cdf(dist_data):
    plt.clf()
    plt.figure(figsize=(4.5, 3.5))

    x, y = sorted(dist_data), np.arange(len(dist_data)) / len(dist_data)
    plt.plot(x, y, color='red', linewidth = 1)
    plt.ylabel('CDF')
    plt.xlabel('Hamming Distance')
    plt.grid()
    #plt.savefig(opt.expname + '/hamming_dist_distribution.png')
    plt.savefig(f"{str_project_root_path}/result/plot/hamming_dist/protected/hamming_dist_distr.pdf", format='pdf')   

if __name__ == "__main__":

    plt.switch_backend('agg')
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    # Training settings
    parser = argparse.ArgumentParser(description='generative adversarial perturbations')
    parser.add_argument('--dataVal', type=str, default='~/autodl-tmp/gap/classification/datasets_logo_181/test',
                    help='data val root')
    parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--expname', type=str, default='metrics_out/hamming_dist_siamese', help='experiment name, output folder')
    parser.add_argument('--checkpoint', type=str, default='siamese_mag_10/netG_model_epoch_11_foolrat_99.73118279569893.pth', help='path to starting checkpoint')
    parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')

    g_opt = parser.parse_args()
    gpulist = [int(i) for i in g_opt.gpu_ids.split(',')]
    n_gpu = len(gpulist)
    print('Running with n_gpu: ', n_gpu)

    # if not torch.cuda.is_available():
    #     raise Exception("No GPU found.")

    # make directories
    # if not os.path.exists(opt.expname):
    #     os.mkdir(opt.expname)
    if not os.path.exists(f"{str_project_root_path}/data/protected/org/"):
        os.makedirs(f"{str_project_root_path}/data/protected/org/")

    if not os.path.exists(f"{str_project_root_path}/data/protected/scaled_org/"):
        os.makedirs(f"{str_project_root_path}/data/protected/scaled_org/")

    if not os.path.exists(f"{str_project_root_path}/data/protected/pertubated/"):
        os.makedirs(f"{str_project_root_path}/data/protected/pertubated/")
    
    # if not os.path.exists(f"{str_project_root_path}/result/plot/hamming_dist/protected/"):
    #     os.makedirs(f"{str_project_root_path}/result/plot/hamming_dist/protected/")
    
    #create_pertubated_images('train', g_opt)
    create_pertubated_images('test', g_opt)
    exit()
    dist_data :list[str, float] = calculate_p_hash()

    with open(f"{str_project_root_path}/result/hamming_dist/protected.csv", 'w') as f_write:
        f_write.write('img.path,distance')
        for str_img_path, f_distance in dist_data:
            f_write.write(f"{os.path.basename(str_img_path)},{f_distance:.2f}\n")
    
    plot_cdf([dist[1] for dist in dist_data])
    # plot_distribution_graph(freq_count)