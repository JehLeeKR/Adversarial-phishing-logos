import numpy as np
import torch
import os
import torchfcn
import cityscapes_fcn8s
from .base_model import BaseModel
from ..utils import image_transform as it
from ..utils import util
from . import generators
from torch.autograd import Variable
from collections import OrderedDict
import torchvision


class MainClfModel(BaseModel):
    def name(self):
        return 'MainClfModel-{}'.format(self.generator.name)

    def move_cuda(self):
        self.criterion_mse = torch.nn.MSELoss().cuda(self.gpulist[0])
        self.criterion_pre = torch.nn.CrossEntropyLoss().cuda(self.gpulist[0])

    def train(self, isTrain):
        self.isTrain = isTrain
        if isTrain is False:
            self.generator.eval()
            self.generator.volatile = True
            self.lbl_trues = []
            self.lbl_preds = []
            self.losses = 0
            if self.isUniversal:
                self.noise = Variable(self.noise_cpu, volatile=True).cuda(self.gpulist[1])
            self.real = Variable(self.real_shape, volatile=True).cuda(self.gpulist[0])
        else:
            self.generator.train()
            self.generator.volatile = False
            if self.isUniversal:
                self.noise = Variable(self.noise_cpu).cuda(self.gpulist[1])
            self.real = Variable(self.real_shape, requires_grad=False).cuda(self.gpulist[0])

    '''
    self.task
    0 for img_dependent untargeted
    1 for img_dependent targeted
    2 for universal untargeted
    3 for universal targeted

    self.metric
    0 for success_rate
    2 for mean_iu
    '''

    def initialize(self, args, n_class):
        BaseModel.initialize(self, args)

        self.gpulist = args.gpu_ids
        self.task = args.task
        self.metric_type = args.metric
        self.dataset = args.dataset
        self.alpha = args.alpha  # coefficient for two losses
        self.mapping = np.random.permutation(n_class)

        self.isTarget = self.task == 1 or self.task == 3
        self.isUniversal = self.task == 2 or self.task == 3

        self.isTrain = True
        self.n_class = n_class
        self.real_shape = torch.FloatTensor(
            args.batch_size, args.output_nc, 224, 224
        ) if args.foolmodel != 'incv3' else torch.FloatTensor(
            args.batch_size, args.output_nc, 299, 299
        )

        # set up the generator based on the arguments
        self.generator = generators.define(args.input_nc, args.output_nc,
                                           args.ngf, args.generator,
                                           args.norm, args.activation,
                                           args.block, self.gpulist)

        # set up different pretrained model for different dataset
        if opt.foolmodel == 'incv3':
            self.pretrained_model = torchvision.models.inception_v3(pretrained=True).cuda()
        elif opt.foolmodel == 'vgg16':
            self.pretrained_model = torchvision.models.vgg16(pretrained=True).cuda()
        elif opt.foolmodel == 'vgg19':
            self.pretrained_model = torchvision.models.vgg19(pretrained=True).cuda()

        self.pretrained_model = self.pretrained_model.eval()
        self.pretrained_model.volatile = True
        self.pretrained_model = self.pretrained_model.cuda(self.gpulist[0])

        # set up the optimizers
        # add optimizer to scheduler (modify learning rate)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(),
                                           lr=args.lr,
                                           betas=(args.beta1, 0.999))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizerG)
        for optimizer in self.optimizers:
            self.schedulers.append(generators.get_scheduler(optimizer, args))

        # set up the criterions
        self.criterion_mse = torch.nn.MSELoss().cuda(self.gpulist[0])
        self.criterion_pre = torch.nn.CrossEntropyLoss().cuda(self.gpulist[0])

        if self.isUniversal:
            # set noise for universal tasks
            self.noise_cpu = torch.rand(
                args.batch_size, args.input_nc,
                args.resolution / 2, args.resolution
            )
            self.noise_cpu.mul_(2.0).add_(-1.0)
            self.eps = args.eps
        else:
            if args.dataset == 'pascalvoc':
                self.eps = args.eps
            elif args.dataset == 'cityscapes':
                self.eps = args.eps

        if self.isTarget:
            # set the target for the cityscapes universal targeted task
            self.target = torch.LongTensor(args.batch_size)
            self.target = Variable(self.target, requires_grad=False)
            self.target = args.target

        # self.move_cuda()

    def set_input(self, input):
        self.real_cpu, self.class_label = input
        self.real.data.resize_(self.real_cpu.size()).copy_(self.real_cpu)

    def forward(self):
        if self.isUniversal is False:
            self.real = self.real.cuda(self.gpulist[1])
            self.delta = self.generator(self.real).cuda(self.gpulist[0])
            self.real = self.real.cuda(self.gpulist[0])
            self.delta = it.inf_norm_adjust(self.delta, self.eps / 128.0)
        else:
            self.delta = self.generator(self.noise).cuda(self.gpulist[0])
            self.delta = it.inf_norm_adjust(self.delta, self.eps / 128.0)

        # crop the delta
        _, _, w, h = self.real.size()
        self.delta = self.delta[:, :, 0: w, 0: h]

        self.fake = torch.add(self.real, self.delta)
        self.fake_cpu = self.fake.cpu()
        for fake_img in self.fake.data:
            fake_img = it.untransform(fake_img, None, 1, self.dataset)
        for fake_img in self.fake.data:
            fake_img = it.transform(fake_img, None, 2, self.dataset)
        self.score = self.pretrained_model(self.fake)
        self.lbl_pred = self.score.data.max(1)[1].cpu().numpy()[:, :, :]

        if self.isTarget is False:
            self.loss = -1 * torch.log(self.criterion_pre(self.score, Variable(self.class_label.cuda())))
            self.lbl_true = self.class_label.numpy()
        else:
            self.loss = torch.log(self.criterion_pre(self.score, self.target))
            self.lbl_true = self.target.data.cpu().numpy()

        if self.isTrain is False:
            for lt, lp in zip(self.lbl_true, self.lbl_pred):
                self.lbl_trues.append(lt)
                self.lbl_preds.append(lp)
            self.losses += self.loss.data[0]

    def backward(self):
        self.loss.backward()
        self.optimizerG.step()

    def optimize_parameters(self):
        self.forward()
        if self.isTrain:
            self.generator.zero_grad()
            self.backward()

    def get_loss(self):
        loss = -1 * torch.log(self.criterion_pre(self.score, Variable(self.class_label.cuda())))
        self.tmp_label = it.do_mapping(self.class_label, self.mapping, self.n_class)
        loss += 1.0 * torch.log(self.criterion_pre(self.score, Variable(self.tmp_label.cuda())))
        return loss
