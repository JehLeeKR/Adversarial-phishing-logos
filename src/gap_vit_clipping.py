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
import time

plt.switch_backend('agg')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='generative adversarial perturbations')
parser.add_argument('--dataTrain', type=str, default='~/autodl-tmp/gap/classification/datasets_logo_181/train',
                    help='data train root')
parser.add_argument('--dataVal', type=str, default='~/autodl-tmp/gap/classification/datasets_logo_181/test',
                    help='data val root')
parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIter', type=int, default=750, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=100
                    , help='Iterations in each Epoch')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--expname', type=str, default='vit_mag_10_customize_clipping', help='experiment name, output folder')
parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='vit',
                    help='model to fool: vit, swin, or siamese')
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test" or "test_fooling_ratio"')
parser.add_argument('--perturbation_type', type=str, default='imdep', help='"imdep" (image dependent)')
parser.add_argument('--target', type=int, default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--threshold_discriminator', help='threshold used by the discriminator for classification', type=float, default=0.95)
parser.add_argument('--threshold_target', help='target threshold', type=float, default=0.50)
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)
parser.add_argument('--loss', help='loss function: cross_entropy or modified_cross_entropy', type=str, default='modified_cross_entropy')

opt = parser.parse_args()

if not torch.cuda.is_available():
    raise Exception("No GPU found.")

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
test_strictly_fooling_history = []
best_fooling = 0
itr_accum = 0
test_threshold = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
test_threshold_count = {0.95 : 0, 0.90 : 0, 0.85 : 0, 0.80 : 0, 0.75 : 0, 0.70 : 0, 0.65 : 0, 0.60 : 0, 0.55 : 0, 0.50 : 0, 0.45 : 0, 0.40 : 0, 0.35 : 0, 0.30 : 0, 0.25 : 0, 0.20 : 0, 0.15 : 0, 0.10 : 0, 0.05 : 0}

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIter = opt.MaxIter
MaxIterTest = opt.MaxIterTest
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)
clipping = opt.output_clipping
loss_func = opt.loss

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

print('===> Loading datasets')
train_annotation_path = './classification/train_data.txt'
test_annotation_path = './classification/test_data.txt'
path_prefix = './classification'

with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(test_annotation_path, encoding='utf-8') as f:
    test_lines = f.readlines()
    
if opt.mode == 'train':    
    train_set = DataGenerator(train_lines, input_shape, True, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
    training_data_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batchSize, num_workers=opt.threads)
        
test_set = DataGenerator(test_lines, input_shape, False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

discriminator_type = opt.foolmodel
discriminator = Discriminator(discriminator_type)

pretrained_discriminator = discriminator.model.cuda(gpulist[0])

pretrained_discriminator.eval()
pretrained_discriminator.volatile = True

# magnitude
mag_in = opt.mag_in

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

# setup optimizer
if opt.optimizer == 'adam':
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == 'sgd':
    optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)


if loss_func == 'cross_entropy':
    criterion_pre = nn.CrossEntropyLoss()
else:
    criterion_pre = CrossEntropyLossWithThreshold(opt.threshold_target)
criterion_pre = criterion_pre.cuda(gpulist[0])


def train(epoch):
    netG.train()
    global itr_accum
    global optimizerG

    for itr, (image, _) in enumerate(training_data_loader, 1):
        start = time.perf_counter()
        if itr > MaxIter:
            break

        if opt.target == -1:
            # least likely class in nontargeted case
            pretrained_label_float = pretrained_discriminator(image.cuda(gpulist[0]))
            target_label = None
            if loss_func == 'cross_entropy':
                _, target_label = torch.min(pretrained_label_float, 1)
            else:
                target_label = torch.div(pretrained_label_float, clipping) 
        else:
            # targeted case
            target_label = torch.LongTensor(image.size(0))
            target_label.fill_(opt.target)
            target_label = target_label.cuda(gpulist[0])

        itr_accum += 1
        if opt.optimizer == 'sgd':
            lr_mult = (itr_accum // 1000) + 1
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr / lr_mult, momentum=0.9)

        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, 'train')

        netG.zero_grad()

        recons = torch.add(image.cuda(gpulist[0]), delta_im.cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        output_pretrained = pretrained_discriminator(recons.cuda(gpulist[0]))
        
        # attempt to get closer to least likely class, or target
        loss = torch.log(criterion_pre(torch.div(output_pretrained, clipping), target_label))

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))
        end = time.perf_counter()
        print(f"Downloaded the tutorial in {end - start:0.4f} seconds")


def test():
    netG.eval()
    correct_recon = 0
    correct_orig = 0
    correct_orig_with_val_above_threshold = 0
    fooled = 0
    total = 0

    for itr, (image, class_label) in enumerate(testing_data_loader):
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
        
        recon_val, predicted_recon = torch.max(outputs_recon, 1)
        orig_val, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon == class_label.cuda(gpulist[0])).sum()
        correct_orig += (predicted_orig == class_label.cuda(gpulist[0])).sum()
        correct_orig_with_val_above_threshold += (orig_val >= opt.threshold_discriminator).sum()

        if opt.target == -1:
            # fooled += (predicted_recon != predicted_orig).sum()
            recon_val = recon_val.tolist()
            orig_val = orig_val.tolist()
            for idx, val in enumerate(orig_val):
                if val >= opt.threshold_discriminator and recon_val[idx] < opt.threshold_discriminator:
                    fooled += 1
        else:
            fooled += (predicted_recon == opt.target).sum()

        if itr % 50 == 1:
            print('Images evaluated:', (itr * opt.testBatchSize))
            # undo normalize image color channels
            delta_im_temp = torch.zeros(delta_im.size())
            for c2 in range(3):
                recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                delta_im_temp[:, c2, :, :] = (delta_im[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            if not os.path.exists(opt.expname):
                os.mkdir(opt.expname)

            post_l_inf = (recons - image[0:recons.size(0)]).abs().max() * 255.0
            print("Specified l_inf:", mag_in, "| maximum l_inf of generated perturbations: %.2f" % (post_l_inf.item()))

            torchvision.utils.save_image(recons, opt.expname + '/reconstructed_{}.png'.format(itr))
            torchvision.utils.save_image(image, opt.expname + '/original_{}.png'.format(itr))
            torchvision.utils.save_image(delta_im_temp, opt.expname + '/delta_im_{}.png'.format(itr))
            print('Saved images.')

    test_acc_history.append((100.0 * correct_recon / total).cpu())
    test_fooling_history.append((100.0 * fooled / total))
    test_strictly_fooling_history.append((100.0 * fooled / correct_orig_with_val_above_threshold).cpu())
    print('Accuracy (label) of the pretrained network on reconstructed images: %.2f%%' % (
            100.0 * float(correct_recon) / float(total)))
    print(
        'Accuracy (label) of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    print(
        'Percentage of classification above threshold of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig_with_val_above_threshold) / float(total)))
    print(
        'Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(correct_orig_with_val_above_threshold)))
    # if opt.target == -1:
    #     print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
    # else:
    #     print('Top-1 Target Accuracy: %.2f%%' % (100.0 * float(fooled) / float(total)))
        
def test_fooling_ratio():
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
        
        recon_val, predicted_recon = torch.max(outputs_recon, 1)
        orig_val, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)

        recon_val = recon_val.tolist()
        orig_val = orig_val.tolist()
        for idx, val in enumerate(orig_val):
            for _, threshold_val in enumerate(test_threshold):
                if val >= threshold_val and recon_val[idx] < threshold_val:
                    test_threshold_count[threshold_val] = test_threshold_count[threshold_val] + 1

    for _, threshold_val in enumerate(test_threshold):
        test_threshold_count[threshold_val] = 100.0 * float(test_threshold_count[threshold_val]) / float(total)

def plot_test_fooling_ratio():
    lists = sorted(test_threshold_count.items())# sorted by key, return a list of tuples
    
    for threshold in test_threshold_count:
        print('Threshold:' + str(threshold) + '  Fooling ratio: %.2f%%' % (test_threshold_count[threshold]))

    thresholds, ratios = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(thresholds, ratios, color='green', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='blue', markersize=6)
    plt.title('Testing Fooling Ratios Against Threshold')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Threshold')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.savefig(opt.expname + '/foolrat_threshold_test_customized_loss_clipping_' + str(clipping) + '.png')
    print("Saved plots.")
    
def test_and_plot_both_fooling_ratio():
    test_fooling_ratio()
    lists = sorted(test_threshold_count.items())
    for threshold in test_threshold_count:
        print('Threshold:' + str(threshold) + '  Fooling ratio: %.2f%%' % (test_threshold_count[threshold]))
    thresholds, ratios = zip(*lists)
    
    for threshold in test_threshold_count:
        test_threshold_count[threshold] = 0
    
    opt.checkpoint = 'vit_mag_10_baseline/netG_model_epoch_299_foolrat_69.36881188118812.pth'
    netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
    test_fooling_ratio()
    lists = sorted(test_threshold_count.items())
    for threshold in test_threshold_count:
        print('Threshold:' + str(threshold) + '  Fooling ratio: %.2f%%' % (test_threshold_count[threshold]))
    thresholds_new, ratios_new = zip(*lists)
    
    plt.plot(thresholds, ratios, color='green', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='blue', markersize=3, label='Modified Cross Entropy Loss')
    for a,b in zip(thresholds, ratios): 
        plt.text(a + 0.05, b, '%.2f%%' % (b), fontsize=6)
        
    plt.plot(thresholds_new, ratios_new, color='red', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='red', markersize=3, label='Original Cross Entropy Loss')
    for a,b in zip(thresholds_new, ratios_new): 
        plt.text(a - 0.05, b, '%.2f%%' % (b), fontsize=6)
        
    plt.title('Testing Fooling Ratios Against Threshold')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Threshold')
    # plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.legend(loc='best')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.savefig(opt.expname + '/foolrat_threshold_test_both_loss.png')
    print("Saved plots.")
    
    
    

def normalize_and_scale(delta_im, mode='train'):
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
            mag_in_scaled_c = mag_in / (255.0 * stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im


def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(
        test_strictly_fooling_history[epoch - 1])

    if test_strictly_fooling_history[epoch - 1] > best_fooling:
        best_fooling = test_strictly_fooling_history[epoch - 1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_strictly_fooling_history[epoch - 1], "Best:", best_fooling)


def print_history():
    # plot history for training loss
    if opt.mode == 'train':
        plt.plot(train_loss_history)
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss'], loc='upper right')
        plt.savefig(opt.expname + '/reconstructed_loss_' + opt.mode + '.png')
        plt.clf()

    # plot history for classification testing accuracy and fooling ratio
    plt.plot(test_acc_history)
    plt.title('Model Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Testing Classification Accuracy'], loc='upper right')
    plt.savefig(opt.expname + '/reconstructed_acc_' + opt.mode + '.png')
    plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname + '/reconstructed_foolrat_' + opt.mode + '.png')
    plt.clf()
    
    plt.plot(test_strictly_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname + '/reconstructed_strict_foolrat_' + opt.mode + '.png')
    print("Saved plots.")

if opt.mode == 'train':
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
    #     test()
    #     checkpoint_dict(epoch)
    # print_history()
elif opt.mode == 'test':
    print('Testing...')
    test()
    # print_history()
elif opt.mode == 'test_fooling_ratio':
    print('Testing...')
    test_fooling_ratio()
    plot_test_fooling_ratio()
elif opt.mode == 'test_both_fooling_ratio':
    print('Testing...')
    test_and_plot_both_fooling_ratio()
    
