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
from utils.dataloader import DataGeneratorV2
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
import imagehash
from gap_util import custom_pil_loader
from PIL import Image
from collections import defaultdict
    

def calculate_p_hash(opt, netG, testing_data_loader, gpulist):
    netG.eval()
    total_images = 0
    total_dist = 0
    frequency_count = defaultdict(float)
    dist_data = []
    pair = 1
    for itr, (image, _) in enumerate(testing_data_loader):
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        delta_im_temp = torch.zeros(delta_im.size())
        for c2 in range(3):
            recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        total_images += len(image)

        for i in range(len(image)):
            torchvision.utils.save_image(recons[i], opt.expname + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.expname + '/original.png')
            original_hash = imagehash.phash(Image.open(opt.expname + '/original.png'))
            reconstructed_hash = imagehash.phash(Image.open(opt.expname + '/reconstructed.png'))
            hash_dist = original_hash - reconstructed_hash
            dist_data.append(hash_dist)
            frequency_count[hash_dist] = frequency_count[hash_dist] + 1
            if not os.path.exists(opt.expname + '/hash_dist_' + str(hash_dist) + '/' + str(pair)):
                os.makedirs(opt.expname + '/hash_dist_' + str(hash_dist) + '/' + str(pair))
            torchvision.utils.save_image(recons[i], opt.expname + '/hash_dist_' + str(hash_dist) + '/' + str(pair) + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.expname + '/hash_dist_' + str(hash_dist) + '/' + str(pair) + '/original.png')
            pair += 1
            total_dist += hash_dist
    
    print('Average hamming distance using perceptual hashing: %.2f' % (float(total_dist) / float(total_images)))
    for dist in frequency_count:
        frequency_count[dist] = float(frequency_count[dist]) / float(total_images)
        print('Hamming distance: ' + str(dist) + ', Frequency: ' +  '%.2f%%' % (100 * frequency_count[dist]))
    return frequency_count, dist_data
        
def normalize_and_scale(delta_im, opt, mode='train', gpulist=None):
    delta_im = delta_im + 1  # now 0..2
    delta_im = delta_im * 0.5  # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:, c, :, :] = (delta_im[:, c, :, :].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(len(delta_im)):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
            mag_in_scaled_c = opt.mag_in / (255.0 * stddev_arr[ci])
            gpu_id = gpulist[0]
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
    plt.title('Hamming Distance between Original Image and Perturbed Image')
    plt.ylabel('Percentage')
    plt.xlabel('Hamming Distance')
    plt.savefig(opt.expname + '/hamming_dist_distribution.png')
    print("Saved plots.")

def plot_cdf(dist_data):
    x, y = sorted(dist_data), np.arange(len(dist_data)) / len(dist_data)
    plt.plot(x, y, color='red', linewidth = 1)
    plt.ylabel('CDF')
    plt.xlabel('Hamming Distance')
    plt.grid()
    plt.savefig(opt.expname + '/hamming_dist_distribution.png')
    
    

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

    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found.")

    # make directories
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    gpulist = [int(i) for i in opt.gpu_ids.split(',')]
    n_gpu = len(gpulist)
    print('Running with n_gpu: ', n_gpu)

    # define normalization means and stddevs
    model_dimension = 256
    center_crop = 224
    input_shape = [224, 224]

    img_size = 128
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean_arr = [0.5, 0.5, 0.5]
    stddev_arr = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)

    data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        normalize,
    ])

    print('===> Loading datasets')
    test_annotation_path = './classification/test_data.txt'
    path_prefix = './classification'

    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
        
    test_set = DataGeneratorV2(test_lines, path_prefix, transform=data_transform, is_test=True)
    testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

    print('===> Building model')

    # will use model paralellism if more than one gpu specified
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

    # resume from checkpoint if specified
    if opt.checkpoint:
        if os.path.isfile(opt.checkpoint):
            print("=> loading checkpoint '{}'".format(opt.checkpoint))
            netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{}'".format(opt.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(opt.checkpoint))
            netG.apply(weights_init)
    else:
        netG.apply(weights_init)
    
    _, dist_data = calculate_p_hash(opt, netG, testing_data_loader, gpulist)
    plot_cdf(dist_data)
    # plot_distribution_graph(freq_count)