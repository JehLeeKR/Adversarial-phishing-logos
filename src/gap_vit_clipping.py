from __future__ import print_function
import argparse
import os
from typing import List, Dict, Tuple, Any
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

def train(epoch: int, opt: argparse.Namespace, netG: nn.Module, pretrained_discriminator: nn.Module,
          optimizerG: optim.Optimizer, criterion_pre: nn.Module, training_data_loader: DataLoader,
          gpulist: List[int], train_loss_history: List[float], itr_accum: int) -> Tuple[List[float], int]:
    """
    Runs a single training epoch for the generator.

    Args:
        epoch (int): The current epoch number.
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        pretrained_discriminator (nn.Module): The pre-trained discriminator to fool.
        optimizerG (optim.Optimizer): The optimizer for the generator.
        criterion_pre (nn.Module): The loss function.
        training_data_loader (DataLoader): The data loader for training data.
        gpulist (List[int]): List of GPU IDs to use.
        train_loss_history (List[float]): A list to store the loss of each iteration.
        itr_accum (int): The accumulated number of iterations.

    Returns:
        Tuple[List[float], int]: The updated training loss history and accumulated iterations.
    """
    netG.train()

    for itr, (image, _) in enumerate(training_data_loader, 1):
        start = time.perf_counter()
        if itr > opt.MaxIter:
            break

        # --- Determine Target Label ---
        if opt.target == -1:
            # For untargeted attacks, the target is the least likely class predicted by the discriminator.
            pretrained_label_float = pretrained_discriminator(image.cuda(gpulist[0]))
            target_label: torch.Tensor
            if opt.loss == 'cross_entropy':
                _, target_label = torch.min(pretrained_label_float, 1)
            else:
                target_label = torch.div(pretrained_label_float, opt.output_clipping)
        else:
            # targeted case
            target_label = torch.LongTensor(image.size(0))
            target_label.fill_(opt.target)
            target_label = target_label.cuda(gpulist[0])

        itr_accum += 1
        # Adjust learning rate for SGD optimizer based on accumulated iterations
        if opt.optimizer == 'sgd':
            lr = opt.lr / ((itr_accum // 1000) + 1)
            optimizerG.param_groups[0]['lr'] = lr

        # --- Generate Adversarial Perturbation ---
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'train', gpulist)

        netG.zero_grad()

        # Create the adversarial image by adding the perturbation
        recons = torch.add(image.cuda(gpulist[0]), delta_im.cuda(gpulist[0]))

        # Clamp the adversarial image to the valid range of the original image
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        output_pretrained = pretrained_discriminator(recons.cuda(gpulist[0]))
        
        # attempt to get closer to least likely class, or target
        loss: torch.Tensor = torch.log(criterion_pre(torch.div(output_pretrained, opt.output_clipping), target_label))

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))
        end = time.perf_counter()
        print(f"Iteration finished in {end - start:0.4f} seconds")

    return train_loss_history, itr_accum


def test(opt: argparse.Namespace, netG: nn.Module, pretrained_discriminator: nn.Module,
         testing_data_loader: DataLoader, gpulist: List[int]) -> Tuple[float, float, float]:
    """
    Evaluates the generator on the test set.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        pretrained_discriminator (nn.Module): The pre-trained discriminator.
        testing_data_loader (DataLoader): The data loader for test data.
        gpulist (List[int]): List of GPU IDs to use.

    Returns:
        Tuple[float, float, float]: A tuple containing accuracy, fooling ratio, and strict fooling ratio.
    """
    netG.eval()
    # Initialize counters
    correct_recon = 0
    correct_orig = 0
    correct_orig_with_val_above_threshold = 0
    fooled = 0
    total = 0

    for itr, (image, class_label) in enumerate(testing_data_loader):
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        # Process the perturbation
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        # Create and clamp the adversarial image
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        outputs_recon = pretrained_discriminator(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_discriminator(image.cuda(gpulist[0]))
        
        # Apply softmax with output clipping to get probabilities
        outputs_recon = torch.softmax(torch.div(outputs_recon, opt.output_clipping), dim=-1)
        outputs_orig = torch.softmax(torch.div(outputs_orig, opt.output_clipping), dim=-1)
        
        # --- Calculate Metrics ---
        # type: torch.Tensor, torch.Tensor
        recon_val, predicted_recon = torch.max(outputs_recon, 1) 
        orig_val, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon == class_label.cuda(gpulist[0])).sum()
        correct_orig += (predicted_orig == class_label.cuda(gpulist[0])).sum()
        correct_orig_with_val_above_threshold += (orig_val >= opt.threshold_discriminator).sum()

        if opt.target == -1:
            # A successful fooling occurs if the original confidence was high and the new confidence is low.
            recon_val = recon_val.tolist()
            orig_val = orig_val.tolist()
            for idx, val in enumerate(orig_val):
                if val >= opt.threshold_discriminator and recon_val[idx] < opt.threshold_discriminator:
                    fooled += 1
        else:
            fooled += (predicted_recon == opt.target).sum()

        # Save sample images periodically
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
            print("Specified l_inf:", opt.mag_in, "| maximum l_inf of generated perturbations: %.2f" % (post_l_inf.item()))

            torchvision.utils.save_image(recons, opt.expname + '/reconstructed_{}.png'.format(itr))
            torchvision.utils.save_image(image, opt.expname + '/original_{}.png'.format(itr))
            torchvision.utils.save_image(delta_im_temp, opt.expname + '/delta_im_{}.png'.format(itr))
            print('Saved images.')

    acc: float = (100.0 * correct_recon / total).cpu().item()
    fool_ratio: float = (100.0 * fooled / total)
    strict_fool_ratio: float = (100.0 * fooled / correct_orig_with_val_above_threshold).cpu().item()
    print('\n--- Test Results ---')
    print('Accuracy (label) of the pretrained network on reconstructed images: %.2f%%' % (
            100.0 * float(correct_recon) / float(total)))
    print(
        'Accuracy (label) of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    print(
        'Percentage of classification above threshold of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig_with_val_above_threshold) / float(total)))
    print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(correct_orig_with_val_above_threshold) ))
    return acc, fool_ratio, strict_fool_ratio

def test_fooling_ratio(opt: argparse.Namespace, netG: nn.Module, pretrained_discriminator: nn.Module,
                       testing_data_loader: DataLoader, gpulist: List[int], test_threshold: List[float],
                       test_threshold_count: Dict[float, int]) -> Dict[float, float]:
    """
    Tests the fooling ratio of a generator across multiple confidence thresholds.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        pretrained_discriminator (nn.Module): The pre-trained discriminator.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs to use.
        test_threshold (List[float]): A list of thresholds to evaluate against.
        test_threshold_count (Dict[float, int]): A dictionary to store fooling counts for each threshold.
    """
    netG.eval()
    total = 0

    for itr, (image, class_label) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        outputs_recon = pretrained_discriminator(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_discriminator(image.cuda(gpulist[0]))
        
        outputs_recon = torch.softmax(torch.div(outputs_recon, opt.output_clipping), dim=-1)
        outputs_orig = torch.softmax(torch.div(outputs_orig, opt.output_clipping), dim=-1)
        
        recon_val, predicted_recon = torch.max(outputs_recon, 1)
        orig_val, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)

        recon_val = recon_val.tolist()
        orig_val = orig_val.tolist()
        for idx, val in enumerate(orig_val):
            for _, threshold_val in enumerate(test_threshold):
                if val >= threshold_val and recon_val[idx] < threshold_val:
                    test_threshold_count[threshold_val] = test_threshold_count[threshold_val] + 1
    
    fooling_ratio_results: Dict[float, float] = {}
    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio_results[threshold_val] = 100.0 * float(test_threshold_count[threshold_val]) / float(total)
    
    return fooling_ratio_results

def plot_test_fooling_ratio(opt: argparse.Namespace, test_threshold_count: Dict[float, float]):
    """
    Plots the fooling ratio against different thresholds.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        test_threshold_count (Dict[float, float]): A dictionary mapping thresholds to fooling ratios.
    """
    lists: List[Tuple[float, float]] = sorted(test_threshold_count.items()) # sorted by key, return a list of tuples
    
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
    plt.savefig(opt.expname + '/foolrat_threshold_test_customized_loss_clipping_' + str(opt.output_clipping) + '.png')
    print("Saved plots.")
    
def test_and_plot_both_fooling_ratio(opt: argparse.Namespace, netG: nn.Module, pretrained_discriminator: nn.Module,
                                     testing_data_loader: DataLoader, gpulist: List[int], test_threshold: List[float],
                                     test_threshold_count: Dict[float, int]):
    """
    Tests and plots a comparison of fooling ratios between two different generator models.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        pretrained_discriminator (nn.Module): The pre-trained discriminator.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs to use.
        test_threshold (List[float]): A list of thresholds to evaluate against.
        test_threshold_count (Dict[float, int]): A dictionary to store fooling counts.
    """
    test_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, test_threshold, test_threshold_count)
    lists = sorted(test_threshold_count.items())
    for threshold in test_threshold_count:
        print('Threshold:' + str(threshold) + '  Fooling ratio: %.2f%%' % (test_threshold_count[threshold]))
    thresholds, ratios = zip(*lists)
    
    for threshold in test_threshold_count:
        test_threshold_count[threshold] = 0
    
    opt.checkpoint = 'vit_mag_10_baseline/netG_model_epoch_299_foolrat_69.36881188118812.pth'
    netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
    test_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, test_threshold, test_threshold_count)
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
    
    
    

def normalize_and_scale(delta_im: torch.Tensor, opt: argparse.Namespace, mode: str = 'train',
                        gpulist: List[int] = None) -> torch.Tensor:
    """
    Normalizes and scales the generated perturbation to respect the L-infinity norm constraint.

    Args:
        delta_im (torch.Tensor): The raw perturbation tensor from the generator (values in [-1, 1]).
        opt (argparse.Namespace): Command-line arguments, containing `mag_in`.
        mode (str): The current mode ('train' or 'test').
        gpulist (List[int]): List of GPU IDs.

    Returns:
        torch.Tensor: The processed perturbation, ready to be added to an image.
    """
    # The output of the generator is in [-1, 1]. First, scale it to [0, 1].
    delta_im = delta_im + 1  # now 0..2
    delta_im = delta_im * 0.5  # now 0..1

    # The perturbation is normalized with the same mean and std as the images.
    # This is because the perturbation will be added to a normalized image.
    for c in range(3):
        delta_im[:, c, :, :] = (delta_im[:, c, :, :].clone() - mean_arr[c]) / stddev_arr[c]

    # Enforce the L-infinity norm constraint for each image in the batch.
    bs: int = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(len(delta_im)):
        # The L-inf norm is applied per-channel, as is common practice.
        # `mag_in` is the max perturbation in pixel space (0-255), so it's scaled by the stddev.
        for ci in range(3):
            l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
            mag_in_scaled_c = opt.mag_in / (255.0 * stddev_arr[ci])
            gpu_id = gpulist[0]
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy() if l_inf_channel != 0 else 0)

    return delta_im


def checkpoint_dict(opt: argparse.Namespace, epoch: int, netG: nn.Module, current_fooling_ratio: float,
                    best_fooling: float) -> float:
    """
    Saves a checkpoint of the generator model if the current fooling ratio is the best so far.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        epoch (int): The current epoch number.
        netG (nn.Module): The generator network.
        current_fooling_ratio (float): The fooling ratio from the current epoch's test.
        best_fooling (float): The best fooling ratio seen so far.

    Returns:
        float: The updated best fooling ratio.
    """
    netG.eval()
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"
    net_g_model_out_path: str = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(current_fooling_ratio)

    if current_fooling_ratio > best_fooling:
        best_fooling = current_fooling_ratio
        torch.save(netG.state_dict(), net_g_model_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print(f"No improvement: {current_fooling_ratio:.4f}, Best: {best_fooling:.4f}")
    return best_fooling

def print_history(opt: argparse.Namespace, train_loss_history: List[float], test_acc_history: List[float],
                  test_fooling_history: List[float], strict_fooling_ratio_history: List[float]):
    """
    Plots and saves the training and testing history.
    """
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
    
    plt.plot(strict_fooling_ratio_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname + '/reconstructed_strict_foolrat_' + opt.mode + '.png')
    print("Saved plots.")

def main():
    """
    Main function to run the training and evaluation of the adversarial generator.
    """
    opt: argparse.Namespace = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found.")

    # --- Setup ---
    # Create experiment directory
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    # Set seeds for reproducibility
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)

    # Initialize history lists and counters
    train_loss_history: List[float] = []
    test_acc_history: List[float] = []
    test_fooling_history: List[float] = []
    strict_fooling_ratio_history: List[float] = []
    best_fooling: float = 0.0
    itr_accum: int = 0

    gpulist: List[int] = [int(i) for i in opt.gpu_ids.split(',')]
    print('Running with n_gpu: ', len(gpulist))

    # --- Data Loading ---
    print('===> Loading datasets')
    train_annotation_path = './classification/train_data.txt'
    test_annotation_path = './classification/test_data.txt'
    path_prefix = './classification'

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
        
    if opt.mode == 'train':    
        train_set = DataGenerator(train_lines, [224, 224], True, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
        training_data_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batchSize, num_workers=opt.threads)
            
    test_set = DataGenerator(test_lines, [224, 224], False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
    testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

    # --- Model Initialization ---
    discriminator = Discriminator(opt.foolmodel)
    pretrained_discriminator = discriminator.model.cuda(gpulist[0])
    pretrained_discriminator.eval()
    pretrained_discriminator.volatile = True

    print('===> Building model')
    # Initialize the generator network
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

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

    # --- Optimizer and Loss Function ---
    if opt.optimizer == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optimizer == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

    if opt.loss == 'cross_entropy':
        criterion_pre = nn.CrossEntropyLoss()
    else:
        criterion_pre = CrossEntropyLossWithThreshold(opt.threshold_target)
    criterion_pre = criterion_pre.cuda(gpulist[0])

    # Define thresholds for evaluation modes
    test_threshold = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    test_threshold_count = {t: 0 for t in test_threshold}

    # --- Main Execution Logic ---
    if opt.mode == 'train':
        for epoch in range(1, opt.nEpochs + 1):
            # Run one epoch of training
            train_loss_history, itr_accum = train(epoch, opt, netG, pretrained_discriminator, optimizerG, criterion_pre, training_data_loader, gpulist, train_loss_history, itr_accum)
            # Evaluate on the test set
            acc, fool_ratio, strict_fool_ratio = test(opt, netG, pretrained_discriminator, testing_data_loader, gpulist)
            test_acc_history.append(acc)
            test_fooling_history.append(fool_ratio)
            strict_fooling_ratio_history.append(strict_fool_ratio)
            # Save checkpoint if performance improved
            best_fooling = checkpoint_dict(opt, epoch, netG, strict_fool_ratio, best_fooling)
        print_history(opt, train_loss_history, test_acc_history, test_fooling_history, strict_fooling_ratio_history)
    elif opt.mode == 'test':
        print('Testing...')
        test(opt, netG, pretrained_discriminator, testing_data_loader, gpulist)
    elif opt.mode == 'test_fooling_ratio':
        print('Testing...')
        fooling_ratios = test_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, test_threshold, test_threshold_count.copy())
        plot_test_fooling_ratio(opt, fooling_ratios)
    elif opt.mode == 'test_both_fooling_ratio':
        print('Testing...')
        test_and_plot_both_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, test_threshold, test_threshold_count)

if __name__ == '__main__':
    main()
    
