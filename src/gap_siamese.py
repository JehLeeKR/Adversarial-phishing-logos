from __future__ import print_function
import argparse
import os
import config
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
from gap_util import custom_pil_loader
from phishpedia.siamese_pedia.inference import pred_siamese
from phishpedia.siamese_eval_util import l2_norm
import PIL.Image as Image


plt.switch_backend('agg')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='generative adversarial perturbations')
parser.add_argument('--dataTrain', type=str, default=os.path.join(config.LOGO_DATASET_DIR, 'train'),
                    help='data train root')
parser.add_argument('--dataVal', type=str, default=os.path.join(config.LOGO_DATASET_DIR, 'test'),
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
parser.add_argument('--MaxIter', type=int, default=700, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=100
                    , help='Iterations in each Epoch')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--expname', type=str, default=config.SIAMESE_FEATURES_DIR, help='experiment name, output folder')
parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='siamese',
                    help='model to fool: vit, swin, or siamese')
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test" or "test_fooling_ratio"')
parser.add_argument('--perturbation_type', type=str, default='imdep', help='"imdep" (image dependent)')
parser.add_argument('--target', type=int, default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--threshold_discriminator', help='threshold used by the discriminator for classification', type=float, default=0.81)
parser.add_argument('--siamese_image_path', help='image for siamese testing', type=str, default=os.path.join(config.DATA_DIR, 'siamese_testing/'))

opt = parser.parse_args()

if not torch.cuda.is_available():
    raise Exception("No GPU found.")

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0
test_threshold_count = {}
starting_threshold = 0.61
end_threshold = 1.0
stride = 0.02

threshold = starting_threshold
while threshold < end_threshold:
    test_threshold_count[threshold] = 0
    threshold += stride

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

# define normalization means and stddevs
model_dimension = 256
center_crop = 224
input_shape = [224, 224]

img_size = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
mean_arr = [0.5, 0.5, 0.5]
stddev_arr = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean, std=std)

data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    normalize,
])

print('===> Loading datasets')
train_annotation_path = os.path.join(config.SRC_DIR, 'classification/train_data.txt')
test_annotation_path = os.path.join(config.SRC_DIR, 'classification/test_data.txt')
path_prefix = config.LOGO_DATASET_DIR

with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(test_annotation_path, encoding='utf-8') as f:
    test_lines = f.readlines()
    
if opt.mode == 'train':    
    train_set = DataGeneratorV2(train_lines, path_prefix, transform=data_transform, is_test=False)
    training_data_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batchSize, num_workers=opt.threads)
        
test_set = DataGeneratorV2(test_lines, path_prefix, transform=data_transform, is_test=True)
testing_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=opt.testBatchSize, num_workers=opt.threads)

# with open('siamese_mag_10/preprocess_sim_test.txt', 'rb') as fp:
#     img_sim_map = pickle.load(fp)
#     print('Done loading similarity list')

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
    
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.8)

# criterion_pre = SiameseLoss().cuda(gpulist[0])
criterion_pre = nn.CosineEmbeddingLoss(margin=0.2).cuda(gpulist[0])

def predict(img, threshold, model, logo_feat_list):
    img_feat = pred_siamese(img, model)
    sim_list = logo_feat_list @ img_feat.T
    idx = np.argsort(sim_list)[::-1][:1]
    sim = np.array(sim_list)[idx]
    return sim[0] < threshold

def train(epoch, opt, netG, model, optimizerG, criterion_pre, training_data_loader, gpulist):
    netG.train()
    global itr_accum
    global scheduler

    for itr, (image, label) in enumerate(training_data_loader, 1):
        if itr > opt.MaxIter:
            break

        itr_accum += 1
        if opt.optimizer == 'sgd':
            lr_mult = (itr_accum // 1000) + 1
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr / lr_mult, momentum=0.9)

        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'train')

        netG.zero_grad()

        recons = torch.add(image.cuda(gpulist[0]), delta_im.cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())

        t = torch.ones(len(recons)) * (-1)
        t = t.cuda(gpulist[0])
        label = label.squeeze(1).cuda(gpulist[0])
        # print(label.size())
        # new_feat = []
        # for idx in range(len(recons)):
        #     img_feat = pred_siamese_short(recons[idx]).unsqueeze(0).clone().requires_grad_(True)
        #     new_feat.append(img_feat)
        # new_feat = torch.cat(new_feat, dim=0).cuda(gpulist[0])
        new_feat = model.features(recons)
        new_feat = l2_norm(new_feat)
        # print(new_feat.size())
        loss = criterion_pre(new_feat, label, t)
#         sim_list = []
#         for idx in range(len(recons)):
#             img_feat = torch.tensor(pred_siamese_short(recons[idx]))
#             label_feat = label[idx]
#             sim_list.append(torch.nn.functional.cosine_similarity(img_feat, label_feat.clone().detach()))
        
#         # attempt to get closer to least likely class, or target
#         loss = criterion_pre(sim_list)
#         loss = Variable(loss, requires_grad = True)

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))
    scheduler.step()


def test(opt, netG, model, logo_feat_list, testing_data_loader, gpulist):
    netG.eval()
    fooled = 0
    total = 0

    for itr, (image, sim_list) in enumerate(testing_data_loader):
        new_sim_list = []
        if itr > opt.MaxIterTest:
            break
            
        image = image.cuda(gpulist[0])
        delta_im = netG(image)
        delta_im = normalize_and_scale(delta_im, opt, 'test')

        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(image[:, cii, :, :].min(),
                                                                      image[:, cii, :, :].max())
        
        
        for c2 in range(3):
            recons[:, c2, :, :] = (recons[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            image[:, c2, :, :] = (image[:, c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
            
        for i in range(len(image)):
            if sim_list[i] < opt.threshold_discriminator:
                continue
            total += 1
            torchvision.utils.save_image(recons[i], opt.siamese_image_path + '/reconstructed.png')
            recon_img = Image.open(opt.siamese_image_path + '/reconstructed.png')
            is_fooled = predict(recon_img, opt.threshold_discriminator, model, logo_feat_list)
            if fooled:
                fooled += 1
        # for idx in range(len(image)):
        #     # _, orig_sim = get_max_sim(image[idx])
        #     _, new_sim = get_max_sim(recons[idx])
        #     # orig_sim_list.append(orig_sim)
        #     new_sim_list.append(new_sim)


        # for idx, val in enumerate(sim_list):
        #     if val >= opt.threshold_discriminator:
        #         total += 1
        #         if new_sim_list[idx] < opt.threshold_discriminator:
        #             fooled += 1

    # test_acc_history.append((100.0 * correct_recon / total).cpu())
    test_fooling_history.append((100.0 * fooled / total))
    # test_strictly_fooling_history.append((100.0 * fooled / correct_orig_with_val_above_threshold).cpu())
    print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
        
def normalize_and_scale(delta_im, opt, mode='train'):
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
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                               mag_in_scaled_c / l_inf_channel.cpu().numpy()) if l_inf_channel != 0 else 0

    return delta_im


def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(
        test_fooling_history[epoch - 1])

    if test_fooling_history[epoch - 1] > best_fooling:
        best_fooling = test_fooling_history[epoch - 1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_fooling_history[epoch - 1], "Best:", best_fooling)


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
    # plt.plot(test_acc_history)
    # plt.title('Model Testing Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Testing Classification Accuracy'], loc='upper right')
    # plt.savefig(opt.expname + '/reconstructed_acc_' + opt.mode + '.png')
    # plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname + '/reconstructed_foolrat_' + opt.mode + '.png')
    plt.clf()

from phishpedia.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from collections import OrderedDict
from phishpedia.siamese_pedia.utils import read_list
import config

if opt.mode == 'train':
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=128, zero_head=True)
    weights = torch.load(config.SIAMESE_MODEL_PATH, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(gpulist[0])
    model.eval()

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch, opt, netG, model, optimizerG, criterion_pre, training_data_loader, gpulist)
        print('Testing....')
        test(opt, netG, model, None, testing_data_loader, gpulist) # logo_feat_list not needed for training test
        checkpoint_dict(epoch)
    print_history()
elif opt.mode == 'test':
    print('Testing...')
    # This mode requires a fully setup model and data, which is better handled in eval scripts.
    print("Please use a dedicated evaluation script for testing.")
    # print_history()
