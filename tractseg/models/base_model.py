
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from os.path import join
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.optim import Adam
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from tractseg.libs.lr_scheduler import CosineAnnealingWarmUpRestarts
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    pass
from torch.cuda.amp import autocast, GradScaler
from tractseg.libs import pytorch_utils
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from tractseg.data import dataset_specific_utils

from monai.networks.nets import UNet

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict

from segment_anything import sam_model_registry
from sam_fact_tt_image_encoder import Fact_tt_Sam
from importlib import import_module


class BaseModel:
    def __init__(self, Config, inference=False):
        self.Config = Config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Do not use during inference because uses a lot more memory
        if not inference:
            torch.backends.cudnn.benchmark = True

        if self.Config.NR_CPUS > 0:
            torch.set_num_threads(self.Config.NR_CPUS)

        if self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "single_direction":
            NR_OF_GRADIENTS = self.Config.NR_OF_GRADIENTS
        elif self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "combined":
            self.Config.NR_OF_GRADIENTS = 3 * self.Config.NR_OF_CLASSES
        else:
            self.Config.NR_OF_GRADIENTS = 33

        if self.Config.LOSS_FUNCTION == "soft_sample_dice":
            self.criterion = pytorch_utils.soft_sample_dice
        elif self.Config.LOSS_FUNCTION == "soft_batch_dice":
            self.criterion = pytorch_utils.soft_batch_dice
        elif self.Config.EXPERIMENT_TYPE == "peak_regression":
            if self.Config.LOSS_FUNCTION == "angle_length_loss":
                self.criterion = pytorch_utils.angle_length_loss
            elif self.Config.LOSS_FUNCTION == "angle_loss":
                self.criterion = pytorch_utils.angle_loss
            elif self.Config.LOSS_FUNCTION == "l2_loss":
                self.criterion = pytorch_utils.l2_loss
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            # self.criterion = nn.MSELoss()   # aggregate by mean
            self.criterion = nn.MSELoss(size_average=False, reduce=True)   # aggregate by sum
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        if self.Config.MODEL.lower() == 'swinunetr':
            NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                                   self.Config.MODEL)
            self.net = NetworkClass(img_size=self.Config.INPUT_DIM,in_channels=NR_OF_GRADIENTS, out_channels=self.Config.NR_OF_CLASSES, spatial_dims=int(self.Config.DIM[0]),use_v2=True, feature_size=self.Config.FEATURE_SIZE)
            # if self.Config.WEIGHTS_PATH:
            #     weight = torch.load(self.Config.WEIGHTS_PATH)
            #     self.net.load_from(weights=weight)
            #     print("Using pretrained self-supervied Swin UNETR backbone weights !")
        elif self.Config.MODEL == 'LatentDiffusionModel':
            NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                                   self.Config.MODEL)
            self.net = NetworkClass(in_channels=NR_OF_GRADIENTS, out_channels=self.Config.NR_OF_CLASSES)
            if self.Config.WEIGHTS_PATH:
                weight = torch.load(self.Config.WEIGHTS_PATH)
                self.net.load_from(weights=weight)
                print("Using pretrained self-supervied Swin UNETR backbone weights !")
        elif self.Config.MODEL == 'MASAM':
            sam, img_embedding_size = sam_model_registry[self.Config.vit_name](image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE else self.Config.RESIZE,
                                                                num_classes=self.Config.NR_OF_CLASSES-1,
                                                                checkpoint=self.Config.WEIGHTS_PATH if self.Config.RESUME_TRAINING==False else None, in_chans=9, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
            pkg = import_module(self.Config.module)
            self.net = pkg.Fact_tt_Sam(sam, self.Config.rank, s=self.Config.scale)

        elif self.Config.MODEL.lower() == 'monai_unet':
            self.net = UNet(in_channels=NR_OF_GRADIENTS, out_channels=self.Config.NR_OF_CLASSES, spatial_dims=int(self.Config.DIM[0]), channels=(4, 8, 16), strides=(2, 2))
        else:
            NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                                   self.Config.MODEL)
            self.net = NetworkClass(n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES,
                                    n_filt=self.Config.UNET_NR_FILT, batchnorm=self.Config.BATCH_NORM,
                                    dropout=self.Config.USE_DROPOUT, upsample=self.Config.UPSAMPLE_TYPE)

        # Print the number of trainable and total parameters
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total parameters: {exp_utils.sizeof_number(total_params)}")
        print(f"Trainable parameters: {exp_utils.sizeof_number(trainable_params)}")

        self.net = self.net.to(self.device)

        # MultiGPU setup
        # (Not really faster (max 10% speedup): GPU and CPU utility low)
        # nr_gpus = torch.cuda.device_count()
        # exp_utils.print_and_save(self.Config.EXP_PATH, "nr of gpus: {}".format(nr_gpus))
        # self.net = nn.DataParallel(self.net)

        if self.Config.COMPILE:
            self.net = torch.compile(self.net, dynamic=False)
        if torch.cuda.device_count() > 1 and self.Config.USE_DP:
            print(f'Using DataParallel across {torch.cuda.device_count()} GPUs')
            self.net = nn.DataParallel(self.net)
        
        if self.Config.OPTIMIZER == "AdamW":
            self.optimizer = AdamW(self.net.parameters(), lr=self.Config.LEARNING_RATE,
                                   weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(self.net.parameters(), lr=self.Config.LEARNING_RATE,
                                    weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(self.net.parameters(), lr=self.Config.LEARNING_RATE,
                                  weight_decay=self.Config.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")

        if APEX_AVAILABLE and self.Config.FP16:
            # Use O0 to disable fp16 (might be a little faster on TitanX)
            # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, verbosity=0, opt_level="O1")
            if not inference:
                print("INFO: Using fp16 training")
        else:
            if not inference:
                print("INFO: Did not find APEX, defaulting to fp32 training")

        if APEX_AVAILABLE and self.Config.FP16:
            self.scaler = GradScaler()

        if self.Config.LR_SCHEDULE:
            if self.Config.LR_SCHEDULE_TYPE == "ReduceLROnPlateau":
                self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode=self.Config.LR_SCHEDULE_MODE,
                                                            patience=self.Config.LR_SCHEDULE_PATIENCE)
            elif self.Config.LR_SCHEDULE_TYPE == "CosineAnnealingLR":
                num_epochs = self.Config.NUM_EPOCHS #self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
                gamma = self.Config.LR_GAMMA
                warmup = int(num_epochs * self.Config.LR_WARMUP_RATIO)
                base_lr = self.Config.LEARNING_RATE 
                T_0 = int(self.Config.LR_CYCLE * num_epochs)
                T_mult = self.Config.LR_T_MULT
                self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)

        if self.Config.LOAD_WEIGHTS:
            exp_utils.print_verbose(self.Config.VERBOSE, "Loading weights ... ({})".format(join(self.Config.EXP_PATH,
                                                                                        self.Config.WEIGHTS_PATH)))
            # load model weights, optimizer state, scheduler state, epoch
            self.load_model(join(self.Config.EXP_PATH, self.Config.WEIGHTS_PATH))

        elif self.Config.RESUME_TRAINING:
            exp_utils.print_verbose(self.Config.VERBOSE, "Loading checkpoints ... ({})".format(join(self.Config.EXP_PATH,
                                                                                        self.Config.WEIGHTS_PATH)))
            # load model weights, optimizer state, scheduler state, epoch
            self.load_checkpoint(join(self.Config.EXP_PATH, self.Config.WEIGHTS_PATH))

    

        # Reset weights of last layer for transfer learning
        # if self.Config.RESET_LAST_LAYER:
        #     self.net.conv_5 = nn.Conv2d(self.Config.UNET_NR_FILT, self.Config.NR_OF_CLASSES, kernel_size=1,
        #                                 stride=1, padding=0, bias=True).to(self.device)
    def train(self, X, y, weight_factor=None, timesteps=None, type=None):
        X = X.contiguous().cuda(self.device, non_blocking=True)  # (bs, slices, features, x, y)
        y = y.contiguous().cuda(self.device, non_blocking=True)  # (bs, slices, classes, x, y)

        if self.Config.DIM == "2D":
            if self.Config.RESIZE:
                bs, slices, features, w, h = X.shape
                X = X.view(bs * slices, features, w, h)
                y = y.view(bs * slices, -1, w, h) 
                X = F.interpolate(X, size=(self.Config.RESIZE, self.Config.RESIZE), mode='bicubic', align_corners=False) # (bs * slices, features, 144, 144) -> (bs * slices, features, 512, 512)
                y = F.interpolate(y, size=(self.Config.RESIZE, self.Config.RESIZE), mode='nearest') # (bs * slices, classes, 144, 144) -> (bs * slices, classes, 512, 512)
                X = X.view(bs, slices, features, self.Config.RESIZE, self.Config.RESIZE)
                y = y.view(bs, slices, -1, self.Config.RESIZE, self.Config.RESIZE)  # keep classes
                
                # final: (bs, slices, classes, x, y)
            
            if self.Config.MODEL == "MASAM":
                bs, slices, classes, w, h = y.shape
                # if MASAM, do not combine batch and slices dimension in input (X).
                y = y.view(bs * slices, -1, w, h) 
            else:
                # if not MASAM, combine batch and slices dimension.
                print(X.shape)
                bs, slices, features, w, h = X.shape
                X = X.view(bs * slices, features, w, h)
                y = y.view(bs * slices, -1, w, h) 
    
        elif self.Config.DIM == "3D":
            if self.Config.RESIZE:
                X = F.interpolate(X, size=(self.Config.RESIZE, self.Config.RESIZE, self.Config.RESIZE), mode='trilinear', align_corners=False)
                y = F.interpolate(y, size=(self.Config.RESIZE, self.Config.RESIZE, self.Config.RESIZE), mode='nearest')
            
            # if self.Config.MODEL == "MASAM":
            #     bs, classes, w, h, d = y.shape
            #     # if MASAM, do not combine batch and slices dimension in input (X).
            #     y = y.transpose(0, 1).contiguous().view(bs * classes, w, h, d)
            # else:
            #     # if not MASAM, combine batch and slices dimension.
            #     print(X.shape)
            #     bs, features, w, h, d = X.shape
            #     X = X.view(bs * slices, features, w, h)
            #     y = y.view(bs * slices, -1, w, h) 

        if type == 'train' or self.Config.DROPOUT_SAMPLING:  
            self.net.train()
        else:
            self.net.train(False)
        self.optimizer.zero_grad() 
        if type == 'train':
            if APEX_AVAILABLE and self.Config.FP16:
                with autocast():
                    if self.Config.MODEL == 'LatentDiffusionModel':
                        outputs = self.net(X, timesteps)
                    elif self.Config.MODEL == 'MASAM':
                        outputs = self.net(batched_input=X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE else self.Config.RESIZE)
                        outputs = outputs['low_res_logits']
                    else:
                        outputs = self.net(X)
                    angle_err = None
                    if weight_factor is not None:
                        if len(y.shape) == 4:  # 2D
                            weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                                y.shape[2], y.shape[3])).cuda()
                        else:  # 3D
                            weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                                y.shape[2], y.shape[3], y.shape[4])).cuda()
                        bundle_mask = y > 0
                        weights[bundle_mask.data] *= weight_factor  # 10

                        if self.Config.EXPERIMENT_TYPE == "peak_regression":
                            loss, angle_err = self.criterion(outputs, y, weights)
                        else:
                            loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
                    else:
                        if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                            loss = self.criterion(F.sigmoid(outputs), y)
                            # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)  # combined loss
                        elif self.Config.LOSS_FUNCTION == "mse":
                            loss = nn.MSELoss()(outputs, y)
                        else:
                            loss = self.criterion(outputs, y)
                            
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.Config.MODEL == 'LatentDiffusionModel':
                    outputs = self.net(X, timesteps)
                elif self.Config.MODEL == 'MASAM':
                    outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE else self.Config.RESIZE)
                    outputs = outputs['low_res_logits']
                else:
                    outputs = self.net(X)
                angle_err = None
                if weight_factor is not None:
                    if len(y.shape) == 4:  # 2D
                        weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
                                            y.shape[2], y.shape[3])).cuda()
                    else:  # 3D
                        weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
                                            y.shape[2], y.shape[3], y.shape[4])).cuda()
                    bundle_mask = y > 0
                    weights[bundle_mask.data] *= weight_factor  # 10

                    # if self.Config.EXPERIMENT_TYPE == "peak_regression":
                    #     loss, angle_err = self.criterion(outputs, y, weights)
                    # else:
                    loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
                else:
                    if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                        loss = self.criterion(F.sigmoid(outputs), y)
                        # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)  # combined loss
                    else:
                        loss = self.criterion(outputs, y)
                loss.backward()

                if self.Config.GRADIENT_CLIP:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.Config.GRADIENT_CLIP)

                self.optimizer.step()
        elif type == 'validate' or type == 'test':
            with torch.no_grad():
                if self.Config.MODEL == 'MASAM':
                    outputs = self.net(batched_input=X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE else self.Config.RESIZE)
                    outputs = outputs['low_res_logits']
                else:
                    outputs = self.net(X)
            angle_err = None
            if weight_factor is not None:
                if len(y.shape) == 4:  # 2D
                    weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
                                        y.shape[2], y.shape[3])).cuda()
                else:  # 3D
                    weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
                                        y.shape[2], y.shape[3], y.shape[4])).cuda()
                bundle_mask = y > 0
                weights[bundle_mask.data] *= weight_factor  
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
            else:
                if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                    loss = self.criterion(F.sigmoid(outputs), y)
                    # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
                else:
                    loss = self.criterion(outputs, y)
        # if self.Config.EXPERIMENT_TYPE == "peak_regression":
        #     f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
        #                                                     max_angle_error=self.Config.PEAK_DICE_THR,
        #                                                     max_length_error=self.Config.PEAK_DICE_LEN_THR)
        # elif self.Config.EXPERIMENT_TYPE == "dm_regression":
        #     f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
        #                                       per_class=True, threshold=self.Config.THRESHOLD)
        # else:
        f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                          threshold=self.Config.THRESHOLD)

        probs = F.sigmoid(outputs) if self.Config.USE_VISLOGGER else None

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        if self.Config.LOG_PER_BUNDLE:
            for i, bundle_name in enumerate(dataset_specific_utils.get_bundle_names(self.Config.CLASSES)[1:]):
                metrics[f'bundle_{bundle_name}_f1'] = f1[i]
        return probs, metrics


    # def test(self, X, y, weight_factor=None):
        
    #     X = X.contiguous().cuda(self.device, non_blocking=True)
    #     y = y.contiguous().cuda(self.device, non_blocking=True)
    
    #     if self.Config.RESIZE:
    #         bs, slices, features, w, h = X.shape
    #         X = X.view(bs * slices, features, w, h)
    #         y = y.view(bs * slices, -1, w, h) 
    #         X = F.interpolate(X, size=(self.Config.RESIZE, self.Config.RESIZE), mode='bicubic', align_corners=False) # (bs * slices, features, 144, 144) -> (bs * slices, features, 512, 512)
    #         y = F.interpolate(y, size=(self.Config.RESIZE, self.Config.RESIZE), mode='nearest') # (bs * slices, classes, 144, 144) -> (bs * slices, classes, 512, 512)
    #         X = X.view(bs, slices, features, self.Config.RESIZE, self.Config.RESIZE)
    #         y = y.view(bs, slices, -1, self.Config.RESIZE, self.Config.RESIZE)  # keep classes
    #         # final: (bs, slices, classes, x, y)
        
    #     if self.Config.MODEL == "MASAM":
    #         bs, slices, classes, w, h = y.shape
    #         # if MASAM, do not combine batch and slices dimension in input (X).
    #         y = y.view(bs * slices, -1, w, h) 
    #     else:
    #         # if not MASAM, combine batch and slices dimension.
    #         bs, slices, features, w, h = X.shape
    #         X = X.view(bs * slices, features, w, h)
    #         y = y.view(bs * slices, -1, w, h) 

    #     if self.Config.DROPOUT_SAMPLING:
    #         self.net.train()
    #     else:
    #         self.net.train(False)
            
    #     with torch.no_grad():
    #         if self.Config.MODEL == 'MASAM':
    #             outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE else self.Config.RESIZE)
    #             outputs = outputs['low_res_logits']
    #         else:
    #             outputs = self.net(X)
                
    #     angle_err = None

    #     if weight_factor is not None:
    #         if len(y.shape) == 4:  # 2D
    #             weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
    #                                   y.shape[2], y.shape[3])).cuda()
    #         else:  # 3D
    #             weights = torch.ones((self.Config.BATCH_SIZE * self.Config.NR_SLICES, self.Config.NR_OF_CLASSES,
    #                                   y.shape[2], y.shape[3], y.shape[4])).cuda()
    #         bundle_mask = y > 0
    #         weights[bundle_mask.data] *= weight_factor
    #         # if self.Config.EXPERIMENT_TYPE == "peak_regression":
    #         #     loss, angle_err = self.criterion(outputs, y, weights)
    #         # else:
    #         loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
    #     else:
    #         if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
    #             loss = self.criterion(F.sigmoid(outputs), y)
    #             # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
    #         else:
    #             loss = self.criterion(outputs, y)

    #     # if self.Config.EXPERIMENT_TYPE == "peak_regression":
    #     #     f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
    #     #                                                     max_angle_error=self.Config.PEAK_DICE_THR,
    #     #                                                     max_length_error=self.Config.PEAK_DICE_LEN_THR)
    #     # elif self.Config.EXPERIMENT_TYPE == "dm_regression":
    #     #     f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
    #     #                                       per_class=True, threshold=self.Config.THRESHOLD)
    #     # else:
    #     f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
    #                                       threshold=self.Config.THRESHOLD)

    #     probs = F.sigmoid(outputs) if self.Config.USE_VISLOGGER else None

    #     metrics = {}
    #     metrics["loss"] = loss.item()
    #     metrics["f1_macro"] = f1
    #     metrics["angle_err"] = angle_err if angle_err is not None else 0

    #     for i, bundle_name in enumerate(dataset_specific_utils.get_bundle_names(self.Config.CLASSES)[1:]):
    #         metrics['f1_macro_tract_' + bundle_name] = f1[i]
    #     return probs, metrics


    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).contiguous().to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
            
        with torch.no_grad():
            if self.Config.MODEL == 'MASAM':
                outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1])
                outputs = outputs['low_res_logits']
            else:
                outputs = self.net(X)  # forward
        probs = F.sigmoid(outputs).detach().cpu().numpy()

        if self.Config.DIM == "2D":
            probs = probs.transpose(0, 2, 3, 1)  # (bs, x, y, classes)
        else:
            probs = probs.transpose(0, 2, 3, 4, 1)  # (bs, x, y, z, classes)
        return probs


    def save_model(self, metrics, epoch_nr, mode="f1"):
        if mode == "f1":
            #max_f1_idx = np.argmax(metrics["f1_macro_validate"])
            max_f1 = np.max(metrics["f1_macro_validate"])
            latest_f1 = metrics["f1_macro_validate"][-1]
            do_save = latest_f1 == max_f1 #and max_f1 > 0 #0.01
        else:
            #min_loss_idx = np.argmin(metrics["loss_validate"])
            min_loss = np.min(metrics["loss_validate"])
            do_save = min_loss == metrics["loss_validate"][-1]

        # saving to network drives takes 5s (to local only 0.5s) -> do not save too often
        if do_save:
            print("  Saving weights...")
            for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep*")):  # remove weights from previous epochs
                os.remove(fl)
            try:
                #Actually is a pkl not a npz
                # pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                #                               unet=self.net)
                pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                                              epoch=epoch_nr,unet=self.net, optimizer=self.optimizer, scheduler=self.scheduler)
            except IOError:
                print("\nERROR: Could not save weights because of IO Error\n")

    def load_model(self, path):
        if self.Config.RESET_LAST_LAYER:
            pytorch_utils.load_checkpoint_selectively(path, unet=self.net)
        else:
            kwargs = pytorch_utils.load_checkpoint(path, unet=self.net)

    def load_checkpoint(self, path):
        if self.Config.RESET_LAST_LAYER:
            pytorch_utils.load_checkpoint_selectively(path, epoch=self.Config.BEST_EPOCH, unet=self.net, optimizer=self.optimizer, scheduler=self.scheduler)
        else:
            kwargs = pytorch_utils.load_checkpoint(path, epoch=self.Config.BEST_EPOCH, unet=self.net, optimizer=self.optimizer, scheduler=self.scheduler)
            self.Config.BEST_EPOCH = kwargs["epoch"]
    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            exp_utils.print_and_save(self.Config.EXP_PATH, "current learning rate: {}".format(param_group['lr']))

