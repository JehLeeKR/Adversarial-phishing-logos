import os

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .nets import get_model_from_name
from utils.callbacks import LossHistory, AccuracyHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   Whether to use Cuda
    #   Can be set to False if there is no GPU
    # ----------------------------------------------------#
    cuda: bool = True
    # ---------------------------------------------------------------------#
    #   distributed     Specifies whether to use single-machine multi-GPU distributed training.
    #                   Terminal commands are only supported on Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs on Ubuntu.
    #                   Windows systems use DP mode by default to call all GPUs and do not support DDP (DistributedDataParallel).
    #   DP模式：
    #       Set             distributed = False
    #       In the terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       Set             distributed = True
    #       In the terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed: bool = False
    # ---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available for multi-GPU in DDP mode.
    # ---------------------------------------------------------------------#
    sync_bn: bool = False
    # ----------------------------------------------------#
    #   When training your own dataset, be sure to modify classes_path
    #   to point to your corresponding classes txt file.
    # ----------------------------------------------------#
    classes_path: str = 'datasets_logo_181/classes.txt'
    # ----------------------------------------------------#
    #   Input image size
    # ----------------------------------------------------#
    input_shape: List[int] = [224, 224]
    # ------------------------------------------------------#
    #   Model type used:
    #   resnet50
    #   vit_b_16
    #   swin_transformer_small
    # ------------------------------------------------------#
    backbone: str = "vit_b_16"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Whether to use the pre-trained weights of the backbone network. Here, the backbone weights are used, so they are loaded during model construction.
    #   If model_path is set, the backbone weights do not need to be loaded, and the value of `pretrained` is meaningless.
    #   If model_path is not set and pretrained = True, only the backbone is loaded to start training.
    #   If model_path is not set and pretrained = False, and Freeze_Train = False, training starts from scratch without a backbone freezing phase.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained: bool = True

    weights_downloaded: bool = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   For downloading weight files, please refer to the README; they can be downloaded from a cloud drive. The model's pre-trained weights are generally applicable to different datasets because the features are generic.
    #   An important part of the model's pre-trained weights is the weights of the backbone feature extraction network, which is used for feature extraction.
    #   Pre-trained weights are necessary in 99% of cases. Without them, the backbone weights are too random, the feature extraction effect is not obvious, and the network training results will not be good.
    #
    #   If training is interrupted, you can set model_path to a weight file in the logs folder to reload the partially trained weights.
    #   At the same time, modify the parameters of the freezing or unfreezing stage below to ensure the continuity of the model's epochs.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   Here, the weights of the entire model are used, so they are loaded in train.py. `pretrained` does not affect the weight loading here.
    #   If you want the model to start training from the backbone's pre-trained weights, set model_path = '' and pretrained = True. In this case, only the backbone is loaded.
    #   If you want the model to train from scratch, set model_path = '' and pretrained = False.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path: str = ''
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed-precision training.
    #               It can reduce VRAM usage by about half and requires PyTorch 1.7.1 or higher.
    # ---------------------------------------------------------------------#
    fp16: bool = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two stages: the freezing stage and the unfreezing stage. The freezing stage is set to meet the training needs of users with insufficient machine performance.
    #   Freezing training requires less VRAM. If the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch, in which case only freezing training is performed.
    #      
    #   Here are some parameter setting suggestions, which trainers can flexibly adjust according to their needs:
    #   (1) Start training from the pre-trained weights of the entire model:
    #       Adam：
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3. (Freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3. (Unfreeze)
    #       SGD：
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 200, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2. (Freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 200, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2. (Unfreeze)
    #       Note: UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Train from scratch:
    #       Adam：
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Unfreeze_batch_size >= 16, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3. (Unfreeze)
    #       SGD：
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Unfreeze_batch_size >= 16, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2. (Unfreeze)
    #       Note: UnFreeze_Epoch should preferably not be less than 300.
    #   (3) batch_size setting:
    #       The larger the better, within the acceptable range of the graphics card. Insufficient VRAM is not related to the dataset size. 
    #       If you get an out of memory error (OOM or cuda out of memory), please reduce the batch_size.
    #       Due to the BatchNorm layer, the minimum batch_size is 2, it cannot be 1.
    #       Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times Unfreeze_batch_size. It is not recommended to set a large gap, as it affects the automatic adjustment of the learning rate.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Freeze stage training parameters
    #   At this time, the model's backbone is frozen, and the feature extraction network does not change.
    #   It occupies less VRAM and only fine-tunes the network.
    #   Init_Epoch          The starting epoch for the current training. Its value can be greater than Freeze_Epoch. For example, if you set:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       It will skip the freezing stage, start directly from epoch 60, and adjust the corresponding learning rate.
    #                       (Used for resuming from a checkpoint)
    #   Freeze_Epoch        The Freeze_Epoch for model freezing training.
    #                       (Ineffective when Freeze_Train=False)
    #   Freeze_batch_size   The batch_size for model freezing training.
    #                       (Ineffective when Freeze_Train=False)
    # ------------------------------------------------------------------#
    Init_Epoch: int = 0
    Freeze_Epoch: int = 50
    Freeze_batch_size: int = 32
    # ------------------------------------------------------------------#
    #   Unfreeze stage training parameters
    #   At this time, the model's backbone is not frozen, and the feature extraction network will change.
    #   It occupies more VRAM, and all parameters of the network will change.
    #   UnFreeze_Epoch          Total training epochs for the model.
    #   Unfreeze_batch_size     The batch_size of the model after unfreezing.
    # ------------------------------------------------------------------#
    UnFreeze_Epoch: int = 200
    Unfreeze_batch_size: int = 32
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freeze training.
    #                   By default, it freezes the backbone for training first, then unfreezes for further training.
    # ------------------------------------------------------------------#
    Freeze_Train: bool = True

    # ------------------------------------------------------------------#
    #   Other training parameters: related to learning rate, optimizer, and learning rate decay.
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate of the model.
    #                   It is recommended to set Init_lr=1e-3 when using the Adam optimizer.
    #                   It is recommended to set Init_lr=1e-2 when using the SGD optimizer.
    #   Min_lr          Minimum learning rate of the model, defaults to 0.01 of the maximum learning rate.
    # ------------------------------------------------------------------#
    Init_lr: float = 1e-2
    Min_lr: float = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer to use, options are 'adam', 'sgd'.
    #                   When using the Adam optimizer, it is recommended to set Init_lr=1e-3.
    #                   When using the SGD optimizer, it is recommended to set Init_lr=1e-2.
    #   momentum        The momentum parameter used inside the optimizer.
    #   weight_decay    Weight decay, can prevent overfitting.
    #                   There might be errors when using the Adam optimizer, it is recommended to set it to 0.
    # ------------------------------------------------------------------#
    optimizer_type: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   The learning rate decay method to use, options are 'step', 'cos'.
    # ------------------------------------------------------------------#
    lr_decay_type: str = "cos"
    # ------------------------------------------------------------------#
    #   save_period     Save weights every `save_period` epochs.
    # ------------------------------------------------------------------#
    save_period: int = 10
    # ------------------------------------------------------------------#
    #   save_dir        The folder to save weights and log files.
    # ------------------------------------------------------------------#
    save_dir: str = 'exp_full_vit_181_sgd'
    # ------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threading to read data.
    #                   Enabling it will speed up data reading but will occupy more memory.
    #                   Computers with less memory can set it to 2 or 0.
    # ------------------------------------------------------------------#
    num_workers: int = 4

    # ------------------------------------------------------#
    #   train_annotation_path   Path to training image paths and labels.
    #   test_annotation_path    Path to validation image paths and labels (using the test set as the validation set).
    # ------------------------------------------------------#
    train_annotation_path: str = "train_data.txt"
    test_annotation_path: str = 'test_data.txt'

    # ------------------------------------------------------#
    #   Set the GPUs to be used for training.
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   Download pre-trained weights
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0 and not weights_downloaded:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # ------------------------------------------------------#
    #   Get classes
    # ------------------------------------------------------#
    class_names: List[str]
    num_classes: int
    class_names, num_classes = get_classes(classes_path)

    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        # ------------------------------------------------------#
        #   For weight files, please see the README, download from Baidu Netdisk.
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load based on the keys of the pre-trained weights and the model.
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key: List[str] = []
        no_load_key: List[str] = []
        temp_dict: Dict[str, Any] = {}
        
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   Show keys that did not match.
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mFriendly reminder: It is normal for the head part not to be loaded. It is an error if the Backbone part is not loaded.\033[0m")

    # ----------------------#
    #   Record Loss
    # ----------------------#
    if local_rank == 0:
        loss_history: LossHistory = LossHistory(save_dir, model, input_shape=input_shape)
        acc_history: AccuracyHistory = AccuracyHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history: Optional[LossHistory] = None
        acc_history: Optional[AccuracyHistory] = None

    # ------------------------------------------------------------------#
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 or higher to correctly use fp16.
    #   Therefore, "could not be resolved" is displayed here for torch 1.2.
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler: Optional[GradScaler] = GradScaler()
    else:
        scaler: Optional[GradScaler] = None

    model_train = model.train()
    # ----------------------------#
    #   Multi-GPU sync Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if cuda:
        if distributed:
            # ----------------------------#
            #   Multi-GPU parallel execution
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=False)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   Read the corresponding txt for the dataset
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines: List[str] = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines: List[str] = f.readlines()
    num_train: int = len(train_lines)
    num_val: int = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
    # ---------------------------------------------------------#
    #   Total training epochs refers to the total number of times the entire dataset is traversed.
    #   Total training steps refers to the total number of gradient descents.
    #   Each training epoch contains several training steps, and one gradient descent is performed in each training step.
    #   Here, only the minimum training epochs are recommended, with no upper limit. Only the unfreezing part is considered in the calculation.
    # ----------------------------------------------------------#
    wanted_step: float = 3e4 if optimizer_type == "sgd" else 1e4
    total_step: int = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch: float = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training steps to more than %d.\033[0m" % (
            optimizer_type, wanted_step))
        print(
            "\033[1;33;44m[Warning] The total amount of training data for this run is %d, Unfreeze_batch_size is %d, a total of %d epochs are trained, and the calculated total training steps are %d.\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training steps are %d, which is less than the recommended total steps %d, it is recommended to set the total epochs to %d.\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    #   The features of the backbone feature extraction network are generic. Freezing training can speed up training.
    #   It can also prevent the weights from being destroyed in the early stage of training.
    #   Init_Epoch is the starting epoch.
    #   Freeze_Epoch is the epoch for freezing training.
    #   UnFreeze_Epoch is the total training epochs.
    #   If you get an OOM or insufficient VRAM error, please reduce the Batch_size.
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag: bool = False
        # ------------------------------------#
        #   Freeze a certain part for training
        # ------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        # -------------------------------------------------------------------#
        #   If not freezing for training, directly set batch_size to Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size: int = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Determine the current batch_size and adaptively adjust the learning rate
        # -------------------------------------------------------------------#
        nbs: int = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            nbs = 256
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        # ---------------------------------------#
        #   Get the formula for learning rate decay
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   Determine the length of each epoch
        # ---------------------------------------#
        epoch_step: int = num_train // batch_size
        epoch_step_val: int = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = DataGenerator(train_lines, input_shape, True)
        val_dataset = DataGenerator(val_lines, input_shape, False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
        # ---------------------------------------#
        #   Start model training
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   If the model has a frozen learning part,
            #   then unfreeze and set parameters.
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size: int = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   Determine the current batch_size and adaptively adjust the learning rate
                # -------------------------------------------------------------------#
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                    nbs = 256
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   Get the formula for learning rate decay
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.Unfreeze_backbone()

                epoch_step: int = num_train // batch_size
                epoch_step_val: int = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, acc_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          UnFreeze_Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank)

        loss_history.writer.close()
        acc_history.writer.close()
