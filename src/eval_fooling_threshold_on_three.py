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
from phishpedia.src.siamese import phishpedia_config
from phishpedia.src.siamese_pedia.inference import pred_siamese
from phishpedia.src.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from phishpedia.src.siamese_pedia.utils import brand_converter
from phishpedia.src.siamese_pedia.utils import resolution_alignment
from utils.utils import get_classes
from PIL import Image
from collections import OrderedDict
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
parser.add_argument('--expname', type=str, default='metrics_out/fooling_ratio_siamese', help='experiment name, output folder')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIterTest', type=int, default=500, help='Iterations in each Epoch')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--checkpoint_customize', type=str, default='siamese_mag_10_probability/netG_model_epoch_48_foolrat_99.86868023637557.pth', help='path to starting checkpoint')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--starting_threshold', help='threshold used by the discriminator for classification', type=float, default=0.61)
parser.add_argument('--ending_threshold', help='maximum threshold used by the discriminator for classification', type=float, default=1.0)
parser.add_argument('--stride', help='threshold stride', type=float, default=0.02)
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)
parser.add_argument('--siamese_image_path', help='image for siamese testing', type=str, default='siamese_testing/')
parser.add_argument("--weights_path", help="weights path", default='classification/phishpedia/src/siamese_pedia/finetune_bit.pth.tar')
parser.add_argument("--logo_feature_list", help="logo feature list for phishpedia", type=str, default='classification/phishpedia_data/step_relu/logo_feat_list_224.txt')
parser.add_argument("--brand_list", help="brand list for phishpedia", type=str, default='classification/phishpedia_data/step_relu/brand_list_224.txt')
parser.add_argument("--siamese_weights_path", help="weights path", default='classification/phishpedia/src/siamese_pedia/finetune_bit.pth.tar')

opt = parser.parse_args()

# if not torch.cuda.is_available():
#     raise Exception("No GPU found.")

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

with open(test_annotation_path, encoding='utf-8') as f:
    test_lines = f.readlines()
    
test_set = DataGenerator(test_lines, input_shape, False, autoaugment_flag=False, transform=data_transform, prefix=path_prefix)
testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)
# magnitude
mag_in = opt.mag_in

# will use model paralellism if more than one gpu specified
netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

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

                    
def load_generator():
    print("=> loading checkpoint '{}'".format(opt.checkpoint_customize))
    netG.load_state_dict(torch.load(opt.checkpoint_customize, map_location=lambda storage, loc: storage))
    print("=> loaded checkpoint '{}'".format(opt.checkpoint_customize))

    
    

def predict(img, threshold, model, logo_feat_list):
    img_feat = pred_siamese(img, model)
    sim_list = logo_feat_list @ img_feat.T
    idx = np.argsort(sim_list)[::-1][:3]
    sim = np.array(sim_list)[idx]
    return sim[0], sim[0] < threshold

def test_fooling_ratio_siamese_without_step_relu():
    fooling_ratio = []
    for i in test_threshold:
        test_threshold_count[i] = 0
        tp_count[i] = 0
    load_generator()       
    netG.eval()
    
    logo_feat_list = read_list('classification/phishpedia_data/logo_feat_list_224.txt')
    file_name_list = read_list('classification/phishpedia_data/brand_list_224.txt')
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=181, zero_head=True)

    # Load weights
    weights = torch.load(opt.siamese_weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    for itr, (image, _) in enumerate(testing_data_loader):
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
            
        for c2 in range(3):
                recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        for i in range(len(image)):
            torchvision.utils.save_image(recons[i], opt.siamese_image_path + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.siamese_image_path + '/original.png')
            orig_img = Image.open(opt.siamese_image_path + '/original.png')
            recon_img = Image.open(opt.siamese_image_path + '/reconstructed.png')
            for k in test_threshold:
                _, orig_fooled = predict(orig_img, k, model, logo_feat_list)
                _, recon_fooled = predict(recon_img, k, model, logo_feat_list)
                if not orig_fooled:
                    tp_count[k] += 1
                    if recon_fooled:
                        test_threshold_count[k] += 1
                        print('Fooled...')

    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
    
    # assert len(fooling_ratio) == 5
    # fooling_ratio_latter = read_list('metrics_out/fooling_ratio/siamese_fooling_ratio_without_step_relu.txt')
    # fooling_ratio = fooling_ratio + fooling_ratio_latter
    assert len(fooling_ratio) == 20
    print(fooling_ratio)
    write_list(fooling_ratio, opt.expname + '/siamese_fooling_ratio_224.txt')
    print('Saving siamese...')
    return fooling_ratio

def test_fooling_ratio_siamese():
    fooling_ratio = []
    for i in test_threshold:
        test_threshold_count[i] = 0
        tp_count[i] = 0
    load_generator()       
    netG.eval()
    
    logo_feat_list = read_list(opt.logo_feature_list)
    file_name_list = read_list(opt.brand_list)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=181, zero_head=True)

    # Load weights
    weights = torch.load(opt.siamese_weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    # for k, v in weights.items():
    #         name = k.split('module.')[1]
    #         new_state_dict[name]=v
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
            
        for c2 in range(3):
                recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        for i in range(len(image)):
            torchvision.utils.save_image(recons[i], opt.siamese_image_path + '/reconstructed.png')
            torchvision.utils.save_image(image[i], opt.siamese_image_path + '/original.png')
            orig_img = Image.open(opt.siamese_image_path + '/original.png')
            recon_img = Image.open(opt.siamese_image_path + '/reconstructed.png')
            for k in test_threshold:
                _, orig_fooled = predict(orig_img, k, model, logo_feat_list)
                _, recon_fooled = predict(recon_img, k, model, logo_feat_list)
                if not orig_fooled:
                    tp_count[k] += 1
                    if recon_fooled:
                        test_threshold_count[k] += 1
                        print('Fooled...')

    for _, threshold_val in enumerate(test_threshold):
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
        
    # original_fooling_ratio = read_list('metrics_out/fooling_ratio/siamese_fooling_ratio.txt')
    # fooling_ratio = fooling_ratio + original_fooling_ratio
    
    assert len(fooling_ratio) == 20
    print(fooling_ratio)
    write_list(fooling_ratio, opt.expname + '/siamese_plus_fooling_ratio_224.txt')
    print('Saving siamese...')
    return fooling_ratio

    
                    
def test_fooling_ratio(use_vit):
    fooling_ratio = []
    discriminator = Discriminator()
    if not use_vit:
        discriminator.switch_to_swin()
    pretrained_discriminator = discriminator.model.cuda(gpulist[0])
    pretrained_discriminator.eval()
    pretrained_discriminator.volatile = True
    
    for i in test_threshold_count:
        test_threshold_count[i] = 0
        tp_count[i] = 0
    load_generator()       
    netG.eval()
    total = 0

    for itr, (image, _) in enumerate(testing_data_loader):
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
        fooling_ratio.append(float(test_threshold_count[threshold_val]) / float(tp_count[threshold_val]))
    
    if use_vit:
        # original_fooling_ratio = read_list('metrics_out/fooling_ratio/vit_fooling_ratio.txt')
        # fooling_ratio = fooling_ratio + original_fooling_ratio
        assert len(fooling_ratio) == 20
        write_list(fooling_ratio, opt.expname + '/vit_fooling_ratio.txt')
        print('Saving vit...')
    else:
        # original_fooling_ratio = read_list('metrics_out/fooling_ratio/swin_fooling_ratio.txt')
        # fooling_ratio = fooling_ratio + original_fooling_ratio
        assert len(fooling_ratio) == 20
        write_list(fooling_ratio, opt.expname + '/swin_fooling_ratio.txt')
        print('Saving swin...')
    print(fooling_ratio)
    return fooling_ratio

                    
def plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_fooling_ratio_step_relu, vit_fp_rates, vit_tp_rates, swin_fp_rates, swin_tp_rates, siamese_fp_rates, siamese_tp_rates, siamese_fp_rates_step_relu, siamese_tp_rates_step_relu):
    plt.clf()
    plt.plot(vit_fp_rates, vit_fooling_ratio, color='blue', linewidth = 1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(swin_fp_rates, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(siamese_fp_rates, siamese_fooling_ratio, color='black', linewidth = 1.5, linestyle = 'dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(siamese_fp_rates_step_relu, siamese_fooling_ratio_step_relu, color='red', linewidth = 1.5, linestyle = 'dashdot', marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Fooling Ratio')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.grid()
    plt.xlim(right=1.1)
    plt.savefig(opt.expname + '/fooling_ratios_against_fp.png')
    plt.show()
    plt.clf()
    
    plt.plot(vit_tp_rates, vit_fooling_ratio, color='blue', linewidth = 1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(swin_tp_rates, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(siamese_tp_rates, siamese_fooling_ratio, color='black', linewidth = 1.5, linestyle = 'dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(siamese_tp_rates_step_relu, siamese_fooling_ratio_step_relu, color='red', linewidth = 1.5, linestyle = 'dashdot',
         marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('True Positive Rate')
    plt.ylabel('Fooling Ratio')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(opt.expname + '/fooling_ratios_against_tp.png')
    plt.show()
    plt.clf()
    
    plt.plot(test_threshold, vit_fooling_ratio, color='blue', linewidth = 1.5,
         marker='D', markerfacecolor='blue', markersize=4, label='ViT')
    plt.plot(test_threshold, swin_fooling_ratio, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(test_threshold, siamese_fooling_ratio, color='black', linewidth = 1.5, linestyle = 'dotted',
        marker='v', markerfacecolor='black', markersize=4, label='Siamese')
    plt.plot(test_threshold, siamese_fooling_ratio_step_relu, color='red', linewidth = 1.5, linestyle = 'dashdot',
         marker='x', markerfacecolor='red', markersize=4, label='Siamese++')
    plt.xlabel('Threshold Value')
    plt.ylabel('Fooling Ratio')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(opt.expname + '/fooling_ratios_against_threshold.png')
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
    tp_rate_siamese = read_list('classification/phishpedia_data/tp_rates_224.txt')
    fp_rate_siamese = read_list('classification/phishpedia_data/fp_rates_224.txt')
    
    tp_rate_siamese_with_step_relu = read_list('classification/phishpedia_data/step_relu/tp_rates_224.txt')
    fp_rate_siamese_with_step_relu = read_list('classification/phishpedia_data/step_relu/fp_rates_224.txt')
    
    tp_rate_vit = read_list('metrics_out/tp_fp/tp_rates_vit.txt')
    fp_rate_vit = read_list('metrics_out/tp_fp/fp_rates_vit.txt')
    
    tp_rate_swin = read_list('metrics_out/tp_fp/tp_rates_swin.txt')
    fp_rate_swin = read_list('metrics_out/tp_fp/fp_rates_swin.txt')
    

    import pandas as pd
    if not os.path.exists('./result/Fool/'):
        os.makedirs('./result/Fool/')   

    list_model = ['ViT', 'Swin', 'Siamese']
    for str_model in list_model:
        str_base_dir = f"./metrics_out/fooling_ratio_{str_model.lower()}/"
    
        vit_fooling_ratio = read_list(f'{str_base_dir}/vit_fooling_ratio.txt')
        swin_fooling_ratio = read_list(f'{str_base_dir}swin_fooling_ratio.txt')
        siamese_fooling_ratio = read_list(f'{str_base_dir}/siamese_fooling_ratio_224.txt')
        siamese_plus_fooling_ratio = read_list(f'{str_base_dir}/siamese_plus_fooling_ratio_224.txt')
        

        df_fool = pd.DataFrame({'ViT.TPR':tp_rate_vit, 
                                    'ViT.FPR': fp_rate_vit, 
                                    'ViT.Fooling.Ratio':vit_fooling_ratio,
                                    'Swin.TPR':tp_rate_swin, 
                                    'Swin.FPR': fp_rate_swin, 
                                    'Swin.Fooling.Ratio':swin_fooling_ratio,
                                    'Siamese.TPR':tp_rate_siamese, 
                                    'Siamese.FPR': fp_rate_siamese, 
                                    'Siamese.Fooling.Ratio':siamese_fooling_ratio,
                                    'Siamese++.TPR':tp_rate_siamese_with_step_relu,
                                    'Siamese++.FPR': fp_rate_siamese_with_step_relu, 
                                    'Siamese++.Fooling.Ratio':siamese_plus_fooling_ratio})
         
        
        df_fool.to_csv(f"./result/Fool/{str_model}_Fool.csv")
    exit()

    
    vit_fooling_ratio = read_list('metrics_out/fooling_ratio_vit/vit_fooling_ratio.txt')
    swin_fooling_ratio = read_list('metrics_out/fooling_ratio_vit/swin_fooling_ratio.txt')
    siamese_fooling_ratio = read_list('metrics_out/fooling_ratio_vit/siamese_fooling_ratio_224.txt')
    siamese_plus_fooling_ratio = read_list('metrics_out/fooling_ratio_vit/siamese_plus_fooling_ratio_224.txt')
    
#     print('Testing fooling ratio with siamese++...')
#     siamese_plus_fooling_ratio = test_fooling_ratio_siamese()
#     print('Testing fooling ratio with siamese...')
#     siamese_fooling_ratio = test_fooling_ratio_siamese_without_step_relu()
#     print('Testing fooling ratio with vit...')
#     vit_fooling_ratio = test_fooling_ratio(use_vit=True)
#     print('Testing fooling ratio with swin...')
#     swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    opt.expname = 'metrics_out/fooling_ratio_vit'
    plot_fooling_ratios(vit_fooling_ratio, 
                        swin_fooling_ratio, 
                        siamese_fooling_ratio, 
                        siamese_plus_fooling_ratio, 
                        fp_rate_vit, 
                        tp_rate_vit, 
                        fp_rate_swin, 
                        tp_rate_swin, 
                        fp_rate_siamese, 
                        tp_rate_siamese, 
                        fp_rate_siamese_with_step_relu, 
                        tp_rate_siamese_with_step_relu)
    
    
    


    # df_swin_fool.to_csv('./plots/csv/Swin_Fool.csv')
    # df_siamese_fool.to_csv('./plots/csv/Siamese_Fool.csv')
    # df_siamese_plus_fool.to_csv('./plots/csv/SiamesePlus_Fool.csv')    
    # exit()
    
    
    
    vit_fooling_ratio = read_list('metrics_out/fooling_ratio_swin/vit_fooling_ratio.txt')
    swin_fooling_ratio = read_list('metrics_out/fooling_ratio_swin/swin_fooling_ratio.txt')
    siamese_fooling_ratio = read_list('metrics_out/fooling_ratio_swin/siamese_fooling_ratio_224.txt')
    siamese_plus_fooling_ratio = read_list('metrics_out/fooling_ratio_swin/siamese_plus_fooling_ratio_224.txt')
    
#     print('Testing fooling ratio with siamese++...')
#     siamese_plus_fooling_ratio = test_fooling_ratio_siamese()
#     print('Testing fooling ratio with siamese...')
#     siamese_fooling_ratio = test_fooling_ratio_siamese_without_step_relu()
#     print('Testing fooling ratio with vit...')
#     vit_fooling_ratio = test_fooling_ratio(use_vit=True)
#     print('Testing fooling ratio with swin...')
#     swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    opt.expname = 'metrics_out/fooling_ratio_swin'
    plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_plus_fooling_ratio, fp_rate_vit, tp_rate_vit, fp_rate_swin, tp_rate_swin, fp_rate_siamese, tp_rate_siamese, fp_rate_siamese_with_step_relu, tp_rate_siamese_with_step_relu)
    
    # opt.checkpoint_customize = 'vit_mag_10_customized_clipping/netG_model_epoch_188_foolrat_92.9568099975586.pth'
    # opt.expname = 'metrics_out/fooling_ratio_vit'
    # if not os.path.exists(opt.expname):
    #     os.mkdir(opt.expname)
    
    
#     print('Testing fooling ratio with vit...')
#     vit_fooling_ratio = test_fooling_ratio(use_vit=True)
#     print('Testing fooling ratio with swin...')
#     swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    
    # opt.checkpoint_customize = 'swin_mag_10_customized_clipping/netG_model_epoch_139_foolrat_97.98561096191406.pth'
    # opt.expname = 'metrics_out/fooling_ratio_swin'
    # if not os.path.exists(opt.expname):
    #     os.mkdir(opt.expname)
    
    
    # print('Testing fooling ratio with vit...')
    # vit_fooling_ratio = test_fooling_ratio(use_vit=True)
    # print('Testing fooling ratio with swin...')
    # swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    
    # opt.checkpoint_customize = 'siamese_mag_10_probability/netG_model_epoch_48_foolrat_99.86868023637557.pth'
    # opt.expname = 'metrics_out/fooling_ratio_siamese'
    # if not os.path.exists(opt.expname):
    #     os.mkdir(opt.expname)
    vit_fooling_ratio = read_list('metrics_out/fooling_ratio_siamese/vit_fooling_ratio.txt')
    swin_fooling_ratio = read_list('metrics_out/fooling_ratio_siamese/swin_fooling_ratio.txt')
    siamese_fooling_ratio = read_list('metrics_out/fooling_ratio_siamese/siamese_fooling_ratio_224.txt')
    siamese_plus_fooling_ratio = read_list('metrics_out/fooling_ratio_siamese/siamese_plus_fooling_ratio_224.txt')
    
#     print('Testing fooling ratio with siamese++...')
#     siamese_plus_fooling_ratio = test_fooling_ratio_siamese()
#     print('Testing fooling ratio with siamese...')
#     siamese_fooling_ratio = test_fooling_ratio_siamese_without_step_relu()
#     print('Testing fooling ratio with vit...')
#     vit_fooling_ratio = test_fooling_ratio(use_vit=True)
#     print('Testing fooling ratio with swin...')
#     swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    opt.expname = 'metrics_out/fooling_ratio_siamese'
    plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_plus_fooling_ratio, fp_rate_vit, tp_rate_vit, fp_rate_swin, tp_rate_swin, fp_rate_siamese, tp_rate_siamese, fp_rate_siamese_with_step_relu, tp_rate_siamese_with_step_relu)
    # print('Testing fooling ratio with vit...')
    # vit_fooling_ratio = test_fooling_ratio(use_vit=True)
    # print('Testing fooling ratio with swin...')
    # swin_fooling_ratio = test_fooling_ratio(use_vit=False)
    
    
    # print('Testing fooling ratio with siamese++...')
    # siamese_plus_fooling_ratio = test_fooling_ratio_siamese()
    # print('Testing fooling ratio with siamese...')
    # siamese_fooling_ratio = test_fooling_ratio_siamese_without_step_relu()
    
    # plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_plus_fooling_ratio, fp_rate_vit, tp_rate_vit, fp_rate_swin, tp_rate_swin, fp_rate_siamese, tp_rate_siamese, fp_rate_siamese_with_step_relu, tp_rate_siamese_with_step_relu)
    # vit_fooling_ratio = read_list('metrics_out/fooling_ratio/vit_fooling_ratio_final.txt')
    # swin_fooling_ratio = read_list('metrics_out/fooling_ratio/swin_fooling_ratio_final.txt')
    # print(vit_fooling_ratio)
    # print(swin_fooling_ratio)
    # siamese_fooling_ratio = read_list('metrics_out/fooling_ratio/siamese_fooling_ratio_without_step_relu.txt')
    # siamese_fooling_ratio_with_step_relu = read_list('metrics_out/fooling_ratio/siamese_fooling_ratio_final.txt')
    
    
    # print(tp_rate_vit)
    # print(fp_rate_vit)
    # print(tp_rate_swin)
    # print(fp_rate_swin)
    # print('Plotting...')
    # plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio, siamese_plus_fooling_ratio, fp_rate_vit, tp_rate_vit, fp_rate_swin, tp_rate_swin, fp_rate_siamese, tp_rate_siamese, fp_rate_siamese_with_step_relu, tp_rate_siamese_with_step_relu)
    
    # plot_fooling_ratios(vit_fooling_ratio, swin_fooling_ratio, siamese_fooling_ratio_without_step_relu, fp_rate_vit, tp_rate_vit, fp_rate_swin, tp_rate_swin, fp_rate_siamese, tp_rate_siamese, 'metrics_out/fooling_ratio/fooling_ratios_against_fp_on_three_discriminators_without_step_relu.png')