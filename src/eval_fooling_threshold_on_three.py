from __future__ import print_function
import argparse
import os
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from material.models.generators import ResnetGenerator, weights_init
from utils.dataloader import DataGenerator
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from classification import Discriminator
from phishpedia.siamese_pedia.inference import pred_siamese
from phishpedia.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from PIL import Image
from collections import OrderedDict
import pickle
import pandas as pd

plt.switch_backend('agg')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# Training settings
def get_args():
    """
    Parses and returns command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a generator\'s fooling ratio against multiple models.')
    parser.add_argument('--models', nargs='+', default=['vit', 'swin', 'siamese', 'siamese+'],
                        choices=['vit', 'swin', 'siamese', 'siamese+'], help='Models to evaluate.')
    parser.add_argument('--dataVal', type=str, default='~/autodl-tmp/gap/classification/datasets_logo_181/test',
                        help='Data validation root directory.')
    parser.add_argument('--testBatchSize', type=int, default=16, help='Testing batch size.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader.')
    parser.add_argument('--expname', type=str, default='metrics_out/fooling_ratio_results', help='Experiment name and output folder.')
    parser.add_argument('--mag_in', type=float, default=10.0, help='L-infinity magnitude of perturbation.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--MaxIterTest', type=int, default=500, help='Maximum iterations for testing.')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in the first convolutional layer.')
    parser.add_argument('--checkpoint_customize', type=str, default='siamese_mag_10_probability/netG_model_epoch_48_foolrat_99.86868023637557.pth', help='Path to the generator checkpoint.')
    parser.add_argument('--gpu_ids', help='GPU IDs: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
    parser.add_argument('--starting_threshold', help='Starting threshold for classification.', type=float, default=0.61)
    parser.add_argument('--ending_threshold', help='Maximum threshold for classification.', type=float, default=1.0)
    parser.add_argument('--stride', help='Threshold stride.', type=float, default=0.02)
    parser.add_argument('--output_clipping', help='Output clipping value for ViT/Swin.', type=float, default=2.5)
    parser.add_argument('--siamese_image_path', help='Path to save temporary images for Siamese testing.', type=str, default='siamese_testing/')
    parser.add_argument("--siamese_weights_path", help="Weights path for Siamese models.", default='phishpedia/siamese_pedia/finetune_bit.pth.tar')
    parser.add_argument("--logo_feature_list", help="Logo feature list for Phishpedia.", type=str, default='phishpedia_data/step_relu/logo_feat_list_224.txt')
    parser.add_argument("--brand_list", help="Brand list for Phishpedia.", type=str, default='phishpedia_data/step_relu/brand_list_224.txt')
    return parser.parse_args()



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


class QuantizeRelu(nn.Module):
    """
    A custom ReLU-like layer that quantizes positive values to steps.
    This is used for the Siamese+ model.
    """
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


# Read list to memory
def read_list(file_path: str) -> Any:
    """
    Reads a pickled list from a binary file.

    Args:
        file_path (str): The path to the file.
    Returns:
        The unpickled data.
    """
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data
    

# write list to binary file
def write_list(data: Any, file_path: str) -> None:
    """
    Writes a list to a binary file using pickle.

    Args:
        data: The data to be written.
        file_path (str): The path to the file.
    """
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')


def load_generator(netG: nn.Module, checkpoint_path: str) -> None:
    """
    Loads the generator model from a checkpoint.

    Args:
        netG (nn.Module): The generator model to load weights into.
        checkpoint_path (str): The path to the checkpoint file.
    """
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    netG.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    print("=> loaded checkpoint '{}'".format(checkpoint_path))


def predict(img: Image.Image, threshold: float, model: nn.Module, logo_feat_list: np.ndarray) -> Tuple[float, bool]:
    """
    Makes a prediction using the Siamese model.

    Args:
        img (Image.Image): The input image.
        threshold (float): The similarity threshold for a match.
        model (torch.nn.Module): The Siamese model.
        logo_feat_list (np.ndarray): The list of logo features.

    Returns:
        tuple: A tuple containing:
            - float: The highest similarity score.
            - bool: True if the image is considered a phishing attempt (similarity < threshold), False otherwise.
    """
    img_feat: np.ndarray = pred_siamese(img, model)
    sim_list: np.ndarray = logo_feat_list @ img_feat.T
    idx: np.ndarray = np.argsort(sim_list)[::-1][:3]
    sim: np.ndarray = np.array(sim_list)[idx]
    return sim[0], sim[0] < threshold

def test_fooling_ratio_siamese(opt: argparse.Namespace, netG: nn.Module, testing_data_loader: DataLoader,
                               gpulist: List[int], test_threshold: List[float]) -> List[float]:
    """
    Calculates the fooling ratio for the standard Siamese model.

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs.
        test_threshold (list): A list of threshold values to test.

    Returns:
        list: A list of fooling ratios corresponding to each threshold.
    """
    fooling_ratio: List[float] = []
    test_threshold_count: Dict[float, int] = {t: 0 for t in test_threshold}
    tp_count: Dict[float, int] = {t: 0 for t in test_threshold}
    netG.eval()
    
    # Load reference logo features for the Siamese model
    logo_feat_list: np.ndarray = read_list('classification/phishpedia_data/logo_feat_list_224.txt')
    
    # --- Initialize Siamese Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=181, zero_head=True)

    # Load pre-trained weights
    weights = torch.load(opt.siamese_weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1] # remove 'module.' prefix
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    for itr, (image, _) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        # Create adversarial image
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # Clamp to valid image range
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())
        
        # Undo normalization to save as standard image files for Siamese model prediction
        for c2 in range(3):
                recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        # Process each image in the batch individually
        for i in range(len(image)):
            torchvision.utils.save_image(recons[i], opt.siamese_image_path + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.siamese_image_path + '/original.png')
            orig_img: Image.Image = Image.open(opt.siamese_image_path + '/original.png')
            recon_img: Image.Image = Image.open(opt.siamese_image_path + '/reconstructed.png')
            
            # Evaluate against each threshold
            for k in test_threshold:
                _, orig_is_phish = predict(orig_img, k, model, logo_feat_list)
                _, recon_is_phish = predict(recon_img, k, model, logo_feat_list)
                # If original is correctly identified (not phish)
                if not orig_is_phish:
                    tp_count[k] += 1
                    # And the adversarial one is misclassified as phish
                    if recon_is_phish:
                        test_threshold_count[k] += 1
                        print('Fooled...')

    # Calculate fooling ratio for each threshold
    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
    
    print(fooling_ratio)
    write_list(fooling_ratio, opt.expname + '/siamese_fooling_ratio_224.txt')
    print('Saving siamese...')
    return fooling_ratio

def test_fooling_ratio_siamese_plus(opt: argparse.Namespace, netG: nn.Module, testing_data_loader: DataLoader,
                                    gpulist: List[int], test_threshold: List[float]) -> List[float]:
    """
    Calculates the fooling ratio for the Siamese+ model (with QuantizeRelu).

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs.
        test_threshold (list): A list of threshold values to test.

    Returns:
        list: A list of fooling ratios corresponding to each threshold.
    """
    fooling_ratio: List[float] = []
    test_threshold_count: Dict[float, int] = {t: 0 for t in test_threshold}
    tp_count: Dict[float, int] = {t: 0 for t in test_threshold}
    
    logo_feat_list: np.ndarray = read_list(opt.logo_feature_list)
    
    # --- Initialize Siamese+ Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=181, zero_head=True)

    # Load weights
    weights = torch.load(opt.siamese_weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    # replace relu with defenselayer 
    model.body.block4.unit01.relu = QuantizeRelu()
    model.body.block4.unit02.relu = QuantizeRelu()
    model.body.block4.unit03.relu = QuantizeRelu()
    
    model.to(device)
    model.eval()

    for itr, (image, _) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        # Create adversarial image
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # Clamp to valid image range
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())
        
        # Undo normalization for saving
        for c2 in range(3):
                recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        # Process each image in the batch
        for i in range(len(image)):
            torchvision.utils.save_image(recons[i], opt.siamese_image_path + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.siamese_image_path + '/original.png')
            orig_img: Image.Image = Image.open(opt.siamese_image_path + '/original.png')
            recon_img: Image.Image = Image.open(opt.siamese_image_path + '/reconstructed.png')
            
            for k in test_threshold:
                _, orig_is_phish = predict(orig_img, k, model, logo_feat_list)
                _, recon_is_phish = predict(recon_img, k, model, logo_feat_list)
                if not orig_is_phish:
                    tp_count[k] += 1
                    if recon_is_phish:
                        test_threshold_count[k] += 1
                        print('Fooled...')

    # Calculate fooling ratio
    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
        
    print(fooling_ratio)
    write_list(fooling_ratio, opt.expname + '/siamese_plus_fooling_ratio_224.txt')
    print('Saving siamese...')
    return fooling_ratio

    
def test_fooling_ratio_vision_transformer(opt: argparse.Namespace, netG: nn.Module, testing_data_loader: DataLoader,
                                          gpulist: List[int], test_threshold: List[float], use_vit: bool = True) -> List[float]:
    """
    Calculates the fooling ratio for Vision Transformer based models (ViT or Swin).

    Args:
        opt (argparse.Namespace): Command-line arguments.
        netG (nn.Module): The generator network.
        testing_data_loader (DataLoader): The data loader for the test set.
        gpulist (List[int]): List of GPU IDs.
        test_threshold (list): A list of threshold values to test.
        use_vit (bool): If True, uses ViT. Otherwise, uses Swin Transformer.

    Returns:
        list: A list of fooling ratios corresponding to each threshold.
    """
    fooling_ratio = []
    # Initialize the appropriate discriminator model
    discriminator = Discriminator()
    if not use_vit:
        discriminator.switch_to_swin()
    pretrained_discriminator = discriminator.model.cuda(gpulist[0])
    
    test_threshold_count: Dict[float, int] = {t: 0 for t in test_threshold}
    tp_count: Dict[float, int] = {t: 0 for t in test_threshold}
    netG.eval()
    total: int = 0

    for itr, (image, _) in enumerate(testing_data_loader):
        print('Processing iteration ' + str(itr) + '...')
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test', gpulist)

        # Create adversarial image
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # Clamp to valid image range
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        # Get predictions for original and adversarial images
        outputs_recon = pretrained_discriminator(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_discriminator(image.cuda(gpulist[0]))
        
        # Apply softmax with clipping
        outputs_recon = torch.softmax(torch.div(outputs_recon, opt.output_clipping), dim=-1)
        outputs_orig = torch.softmax(torch.div(outputs_orig, opt.output_clipping), dim=-1)
        
        recon_val, _ = torch.max(outputs_recon, 1) 
        # type: torch.Tensor, torch.Tensor
        orig_val, _ = torch.max(outputs_orig, 1)
        total += image.size(0)

        recon_val_list: List[float] = recon_val.tolist()
        orig_val_list: List[float] = orig_val.tolist()
        for idx, val in enumerate(orig_val_list):
            for _, threshold_val in enumerate(test_threshold):
                if val >= threshold_val and recon_val_list[idx] < threshold_val:
                    test_threshold_count[threshold_val] = test_threshold_count[threshold_val] + 1
                    
                if val >= threshold_val:
                    tp_count[threshold_val] = tp_count[threshold_val] + 1

    # Calculate fooling ratio
    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
    
    # Save results
    if use_vit:
        write_list(fooling_ratio, opt.expname + '/vit_fooling_ratio.txt')
        print('Saving vit...')
    else:
        write_list(fooling_ratio, opt.expname + '/swin_fooling_ratio.txt')
        print('Saving swin...')
    print(fooling_ratio)
    return fooling_ratio

def plot_fooling_ratios(opt: argparse.Namespace, fooling_ratios: Dict[str, List[float]], fp_rates: Dict[str, List[float]],
                        tp_rates: Dict[str, List[float]], test_threshold: List[float]):
    """
    Plots the fooling ratios against False Positive Rate, True Positive Rate, and Threshold.

    Args:
        fooling_ratios (dict): A dictionary of fooling ratios for each model.
        fp_rates (dict): A dictionary of false positive rates for each model.
        tp_rates (dict): A dictionary of true positive rates for each model.
        test_threshold (list): The list of thresholds used.
    """
    plt.clf()
    vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_plus_fooling_ratio = fooling_ratios.values()
    vit_fp_rates, swin_fp_rates, siamese_fp_rates, siamese_plus_fp_rates = fp_rates.values()
    vit_tp_rates, swin_tp_rates, siamese_tp_rates, siamese_plus_tp_rates = tp_rates.values()

    # --- Plot Fooling Ratio vs. False Positive Rate ---
    plt.plot(vit_fp_rates, vit_fooling_ratio, color='blue', linewidth=1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(swin_fp_rates, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(siamese_fp_rates, siamese_fooling_ratio, color='black', linewidth = 1.5, linestyle = 'dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(siamese_plus_fp_rates, siamese_plus_fooling_ratio, color='red', linewidth = 1.5, linestyle = 'dashdot', 
             marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Fooling Ratio')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.grid()
    plt.xlim(right=1.1)
    plt.savefig(opt.expname + '/fooling_ratios_against_fp.png')
    plt.show()
    plt.clf()

    # --- Plot Fooling Ratio vs. True Positive Rate ---
    plt.plot(vit_tp_rates, vit_fooling_ratio, color='blue', linewidth=1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(swin_tp_rates, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(siamese_tp_rates, siamese_fooling_ratio, color='black', linewidth=1.5, linestyle='dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(siamese_plus_tp_rates, siamese_plus_fooling_ratio, color='red', linewidth = 1.5, linestyle = 'dashdot',
         marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('True Positive Rate')
    plt.ylabel('Fooling Ratio')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(opt.expname + '/fooling_ratios_against_tp.png')
    plt.show()
    plt.clf()

    # --- Plot Fooling Ratio vs. Threshold ---
    plt.plot(test_threshold, vit_fooling_ratio, color='blue', linewidth = 1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(test_threshold, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(test_threshold, siamese_fooling_ratio, color='black', linewidth = 1.5, linestyle = 'dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(test_threshold, siamese_plus_fooling_ratio, color='red', linewidth = 1.5, linestyle = 'dashdot',
         marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('Threshold Value')
    plt.ylabel('Fooling Ratio')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(opt.expname + '/fooling_ratios_against_threshold.png')
    plt.show()

def normalize_and_scale(delta_im: torch.Tensor, opt: argparse.Namespace, mode: str, gpulist: List[int]) -> torch.Tensor:
    """
    Normalizes and scales the generated perturbation to respect the L-infinity norm constraint.

    Args:
        delta_im (torch.Tensor): The raw perturbation from the generator.
        opt (argparse.Namespace): Command-line arguments.
        mode (str): 'train' or 'test'.
        gpulist (List[int]): List of GPU IDs.

    Returns:
        torch.Tensor: The processed perturbation.
    """
    # Scale generator output from [-1, 1] to [0, 1]
    delta_im = delta_im + 1  # now 0..2
    delta_im = delta_im * 0.5  # now 0..1

    # Normalize perturbation with the same stats as the images
    for c in range(3):
        delta_im[:, c, :, :] = (delta_im[:, c, :, :].clone() - mean_arr[c]) / stddev_arr[c]

    # Enforce L-infinity norm constraint
    bs = opt.testBatchSize if mode == 'test' else opt.batchSize
    for i in range(len(delta_im)):
        # Apply L-inf norm per channel
        for ci in range(3):
            l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
            mag_in_scaled_c = opt.mag_in / (255.0 * stddev_arr[ci])
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy() if l_inf_channel != 0 else 0)

    return delta_im

if __name__ == "__main__":
    # --- Setup ---
    opt = get_args()
    cudnn.benchmark = True

    # Create output directories
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)
    if not os.path.exists(opt.siamese_image_path):
        os.makedirs(opt.siamese_image_path)
    
    # Set seeds for reproducibility
    torch.cuda.manual_seed(opt.seed)

    # Configure GPUs
    gpulist = [int(i) for i in opt.gpu_ids.split(',')]
    print('Running with n_gpu: ', len(gpulist))

    # --- Initialize Thresholds ---
    test_threshold = []
    threshold = opt.starting_threshold
    while threshold <= opt.ending_threshold:
        test_threshold.append(round(threshold, 2))
        threshold += opt.stride

    test_threshold_count = {t: 0 for t in test_threshold}

    # --- Load Data and Models ---
    print('===> Loading datasets')
    test_annotation_path = './classification/test_data.txt'
    path_prefix = './classification'
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines: List[str] = f.readlines()
    test_set = DataGenerator(test_lines, input_shape, False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
    testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
    load_generator(netG, opt.checkpoint_customize)

    # --- Load TP/FP rates ---
    fp_rates = {
        'vit': read_list('metrics_out/tp_fp/fp_rates_vit.txt'),
        'swin': read_list('metrics_out/tp_fp/fp_rates_swin.txt'),
        'siamese': read_list('classification/phishpedia_data/fp_rates_224.txt'), # Note: these are pre-computed
        'siameseplus': read_list('classification/phishpedia_data/step_relu/fp_rates_224.txt')
    }
    tp_rates = {
        'vit': read_list('metrics_out/tp_fp/tp_rates_vit.txt'),
        'swin': read_list('metrics_out/tp_fp/tp_rates_swin.txt'),
        'siamese': read_list('classification/phishpedia_data/tp_rates_224.txt'),
        'siameseplus': read_list('classification/phishpedia_data/step_relu/tp_rates_224.txt')
    }

    fooling_ratios = {}

    # --- Run evaluations ---
    if 'vit' in opt.models:
        print('Testing fooling ratio with ViT...')
        fooling_ratios['vit'] = test_fooling_ratio_vision_transformer(opt, netG, testing_data_loader, gpulist, test_threshold, use_vit=True)

    if 'swin' in opt.models:
        print('Testing fooling ratio with Swin...')
        fooling_ratios['swin'] = test_fooling_ratio_vision_transformer(opt, netG, testing_data_loader, gpulist, test_threshold, use_vit=False)

    if 'siamese' in opt.models:
        print('Testing fooling ratio with Siamese...')
        fooling_ratios['siamese'] = test_fooling_ratio_siamese(opt, netG, testing_data_loader, gpulist, test_threshold)

    if 'siamese+' in opt.models:
        print('Testing fooling ratio with Siamese++...')
        fooling_ratios['siameseplus'] = test_fooling_ratio_siamese_plus(opt, netG, testing_data_loader, gpulist, test_threshold)

    # --- Plotting results ---
    print('Plotting results...')
    plot_fooling_ratios(opt, fooling_ratios, fp_rates, tp_rates, test_threshold)

    # --- Save results to CSV ---
    output_csv_dir = './result/Fool/'
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)

    df_data: Dict[str, List[float]] = {}
    for model_name in ['ViT', 'Swin', 'Siamese', 'Siamese++']:
        key = model_name.lower().replace('++', 'plus') # Match dict keys
        if key in fooling_ratios:
            df_data[f'{model_name}.TPR'] = tp_rates[key]
            df_data[f'{model_name}.FPR'] = fp_rates[key]
            df_data[f'{model_name}.Fooling.Ratio'] = fooling_ratios[key]

    df_results = pd.DataFrame(df_data)
    output_csv_path = os.path.join(output_csv_dir, "all_models_fooling_data.csv")
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved fooling data to {output_csv_path}")