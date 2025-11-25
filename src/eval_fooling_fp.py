from __future__ import print_function
import argparse
from typing import List, Dict, Any
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

# Define normalization constants in the global scope so they are accessible
# to functions that need them, like normalize_and_scale.
mean_arr: List[float] = [0.485, 0.456, 0.406]
stddev_arr: List[float] = [0.229, 0.224, 0.225]

# Read list to memory
def read_list(file_path: str) -> Any:
    """
    Reads a Python object from a binary file using pickle.

    Args:
        file_path (str): The path to the file.
    Returns:
        Any: The object loaded from the file.
    """
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data
    
# write list to binary file
def write_list(data: Any, file_path: str) -> None:
    """
    Writes a Python object to a binary file using pickle.

    Args:
        data (Any): The Python object to write.
        file_path (str): The path to the file.
    """
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')


def load_generator(netG: nn.Module, opt: argparse.Namespace, is_baseline: bool) -> None:
    """
    Loads a checkpoint for the generator network.

    Args:
        netG (nn.Module): The generator network.
        opt (argparse.Namespace): The command-line arguments.
        is_baseline (bool): True to load the baseline checkpoint, False for the customized one.
    """
    if is_baseline:
        print("=> loading checkpoint '{}'".format(opt.checkpoint_baseline))
        netG.load_state_dict(torch.load(opt.checkpoint_baseline, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(opt.checkpoint_baseline))
    else:
        print("=> loading checkpoint '{}'".format(opt.checkpoint_customize))
        netG.load_state_dict(torch.load(opt.checkpoint_customize, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(opt.checkpoint_customize))
        
def test_fooling_ratio(opt: argparse.Namespace, netG: nn.Module, pretrained_discriminator: nn.Module,
                       testing_data_loader: DataLoader, gpulist: List[int], is_base: bool,
                       test_threshold: List[float]) -> List[float]:
    """
    Tests the fooling ratio of a generator.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        pretrained_discriminator (nn.Module): The pre-trained discriminator.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs to use.
        is_base (bool): Whether this is the baseline generator.
        test_threshold (List[float]): List of thresholds to test against.

    Returns:
        List[float]: The calculated fooling ratio for each threshold.
    """
    # Initialize counters for this test run
    test_threshold_count: Dict[float, int] = {t: 0 for t in test_threshold}
    tp_count: Dict[float, int] = {t: 0 for t in test_threshold}
    fooling_ratio: List[float] = []
    load_generator(netG, opt, is_base)
    netG.eval()
    total: int = 0

    # Iterate over test data
    for itr, (image, class_label) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        # Process the perturbation to respect the L-inf norm constraint
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        # Create the adversarial image by adding the perturbation
        recons = torch.add(image, delta_im[0:image.size(0)])

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        outputs_recon = pretrained_discriminator(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_discriminator(image) # type: torch.Tensor
        # Apply softmax with output clipping to get probabilities
        outputs_recon = torch.softmax(torch.div(outputs_recon, opt.output_clipping), dim=-1)
        outputs_orig = torch.softmax(torch.div(outputs_orig, opt.output_clipping), dim=-1)
        
        recon_val, _ = torch.max(outputs_recon, 1) 
        # type: torch.Tensor, torch.Tensor
        orig_val, _ = torch.max(outputs_orig, 1)
        total += image.size(0)

        recon_val_list: List[float] = recon_val.tolist()
        orig_val_list: List[float] = orig_val.tolist()
        for idx, val in enumerate(orig_val_list):
            # A "fooled" instance is one where the original confidence was above the threshold,
            # but the adversarial confidence is below it.
            for _, threshold_val in enumerate(test_threshold):
                if val >= threshold_val and recon_val[idx] < threshold_val:
                    test_threshold_count[threshold_val] = test_threshold_count[threshold_val] + 1
                    
                if val >= threshold_val:
                    tp_count[threshold_val] = tp_count[threshold_val] + 1

    # The fooling ratio is the number of fooled instances divided by the number of true positives.
    for threshold_val in test_threshold:
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
    return fooling_ratio

def test_fp(opt: argparse.Namespace, discriminator: Discriminator, test_threshold: List[float], fp_test_path: str) -> List[float]:
    """
    Tests the false positive rate of the discriminator.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        discriminator (Discriminator): The discriminator instance.

    Returns:
        List[float]: A list of false positive rates.
    """
    fp_rate_base = discriminator.evaluate_fp(opt.output_clipping, test_threshold, fp_test_path)
    return fp_rate_base


def plot_fooling_ratio_against_fp(fp_rate_base: List[float], fooling_ratio_base: List[float],
                                  fooling_ratio_customized: List[float], test_threshold: List[float]) -> None:
    """
    Plots the fooling ratio against the false positive rate.

    Args:
        fp_rate_base (List[float]): List of false positive rates.
        fooling_ratio_base (List[float]): List of fooling ratios for the baseline model.
        fooling_ratio_customized (List[float]): List of fooling ratios for the customized model.
        test_threshold (List[float]): The list of thresholds used for evaluation.
    """
    for idx, threshold in enumerate(test_threshold):
        print('Threshold: ' + str("%.2f" % threshold) + ', False Positive Rate: ' + str("%.5f" % fp_rate_base[idx]) + ', Baseline Fooling Ratio: ' + str("%.2f" % fooling_ratio_base[idx]) + ', Fooling Ratio of Customized Generator: ' + str("%.2f" % fooling_ratio_customized[idx]))
    
    plt.figure()
    
    # Plot data
    plt.plot(fp_rate_base, fooling_ratio_base, color='blue', linewidth = 1.5, linestyle = 'dashed',
         marker='D', markerfacecolor='blue', markersize=4, label='Generator with Untargeted Training')
    plt.plot(fp_rate_base, fooling_ratio_customized, color='green', linewidth = 1.5,
         marker='s', markerfacecolor='green', markersize=4, label='Generator with Targeted Training')
    # plt.scatter(fp_rate_base, fooling_ratio_base, color='blue', marker='D', label='Generator with Untargeted Training')
    # plt.scatter(fp_rate_base, fooling_ratio_customized, color='green', marker='s', label='Generator with Targeted Training')
    plt.xlabel('False Positive Rate')
    
    # Configure plot
    plt.ylabel('Fooling Ratio')
    plt.xscale('log')
    plt.xlim(right=1.1)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('metrics_out/fooling_ratio_vit/fooling_ratios_comparison.png')
    plt.show()

def plot_fooling_ratio_against_threshold(fooling_ratio_base: List[float], fooling_ratio_customized: List[float],
                                         test_threshold: List[float]) -> None:
    """
    Plots the fooling ratio against the detection threshold.

    Args:
        fooling_ratio_base (List[float]): List of fooling ratios for the baseline model.
        fooling_ratio_customized (List[float]): List of fooling ratios for the customized model.
        test_threshold (List[float]): The list of thresholds used for evaluation.
    """
    plt.figure()
    
    # Plot data
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
                    
def normalize_and_scale(delta_im: torch.Tensor, opt: argparse.Namespace, mode: str = 'train',
                        gpulist: List[int] = None) -> torch.Tensor:
    """
    Normalizes and scales the generated perturbation to respect the L-infinity norm constraint.
    
    Args:
        delta_im (torch.Tensor): The perturbation tensor.
        opt (argparse.Namespace): Command-line arguments.
        mode (str): The mode ('train' or 'test').
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
    bs: int = opt.testBatchSize
    for i in range(len(delta_im)):
        # The L-inf norm is applied per-channel, as is common practice.
        # `mag_in` is the max perturbation in pixel space (0-255), so it's scaled by the stddev.
        for ci in range(3):
            l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
            mag_in_scaled_c = opt.mag_in / (255.0 * stddev_arr[ci])
            # Scale the perturbation channel to respect the L-inf constraint
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy() if l_inf_channel != 0 else 0)

    return delta_im

def main() -> None:
    """
    Main function to run the evaluation.

    This script performs the following steps:
    1.  Parses command-line arguments.
    2.  Sets up the environment, including seeds and GPU settings.
    3.  Loads the test dataset.
    4.  Initializes the target discriminator model.
    5.  Initializes the generator model.
    6.  Loads pre-calculated fooling ratios and false positive rates from files.
        (The code to generate these is commented out but can be run).
    7.  Generates and saves plots comparing the fooling ratios of two generators
        against the false positive rate and the detection threshold.
    """
    opt: argparse.Namespace = parser.parse_args()

    # --- Setup Thresholds ---
    test_threshold: List[float] = []
    threshold: float = opt.starting_threshold
    while threshold <= opt.ending_threshold:
        test_threshold.append(round(threshold, 2))
        threshold += opt.stride

    # --- Environment Setup ---
    if not torch.cuda.is_available():
        raise Exception("No GPU found.")

    # Create experiment directory if it doesn't exist
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    # Set seeds for reproducibility
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)

    gpulist: List[int] = [int(i) for i in opt.gpu_ids.split(',')]

    # --- Data Loading ---
    normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)
    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print('===> Loading datasets')
    test_annotation_path = './classification/test_data.txt'
    path_prefix = './classification'
    fp_test_annotation_path = 'classification/other_brand_logos/'

    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines: List[str] = f.readlines()
    
    test_set = DataGenerator(test_lines, [224, 224], False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
    testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

    # --- Model Initialization ---
    discriminator = Discriminator()
    pretrained_discriminator = discriminator.model.cuda(gpulist[0])
    pretrained_discriminator.eval()
    pretrained_discriminator.volatile = True

    # Initialize the generator network
    netG: nn.Module = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

    # --- Run Evaluation ---
    # The following lines are commented out as the script is intended to plot pre-computed results.
    # To re-generate the results, uncomment these lines.
    # print('Testing fooling ratio with baseline generator...')
    # baseline_fooling_ratios = test_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, is_base=True, test_threshold=test_threshold)
    # print('Testing fooling ratio with customized generator...')
    # customized_fooling_ratios = test_fooling_ratio(opt, netG, pretrained_discriminator, testing_data_loader, gpulist, is_base=False, test_threshold=test_threshold)
    # print('Testing FP rate...')
    # fp_rates = test_fp(opt, discriminator, test_threshold, fp_test_annotation_path)

    # Load pre-computed results for plotting
    baseline_fooling_ratios = read_list('metrics_out/fooling_ratio_vit_old/vit_fooling_ratio_base_final.txt')
    customized_fooling_ratios = read_list('metrics_out/fooling_ratio_vit_old/vit_fooling_ratio_final.txt')
    fp_rates = read_list('metrics_out/tp_fp/fp_rates_vit.txt')

    # --- Plotting ---
    print('Plotting...')
    plot_fooling_ratio_against_fp(fp_rates, baseline_fooling_ratios, customized_fooling_ratios, test_threshold)
    plt.clf()
    plot_fooling_ratio_against_threshold(baseline_fooling_ratios, customized_fooling_ratios, test_threshold)

if __name__ == "__main__":
    main()