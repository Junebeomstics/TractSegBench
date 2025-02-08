
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

from monai.networks.nets import UNet

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def sizeof_number(number, currency=None):
    """
    format values per thousands : K-thousands, M-millions, B-billions. 
    
    parameters:
    -----------
    number is the number you want to format
    currency is the prefix that is displayed if provided (€, $, £...)
    
    """
    currency='' if currency is None else currency + ' '
    for unit in ['','K','M']:
        if abs(number) < 1000.0:
            return f"{currency}{number:6.2f}{unit}"
        number /= 1000.0
    return f"{currency}{number:6.2f}B"

class BaseModel:
    def __init__(self, Config, inference=False):
        self.Config = Config

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
            self.net = NetworkClass(img_size=self.Config.INPUT_DIM,in_channels=NR_OF_GRADIENTS, out_channels=self.Config.NR_OF_CLASSES, spatial_dims=int(self.Config.DIM[0]),use_v2=True)
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
            from segment_anything import sam_model_registry
            from sam_fact_tt_image_encoder import Fact_tt_Sam
            from importlib import import_module
            sam, img_embedding_size = sam_model_registry[self.Config.vit_name](image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE_TO_512 else 512,
                                                                num_classes=self.Config.NR_OF_CLASSES-1,
                                                                checkpoint=self.Config.WEIGHTS_PATH if self.Config.RESUME_TRAINING==False else None, in_chans=9, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
            pkg = import_module(self.Config.module)
            self.net = pkg.Fact_tt_Sam(sam, self.Config.rank, s=self.Config.scale).cuda()

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
        print(f"Total parameters: {sizeof_number(total_params)}")
        print(f"Trainable parameters: {sizeof_number(trainable_params)}")

        if self.Config.compile:
            self.net = torch.compile(self.net)
        # MultiGPU setup
        # (Not really faster (max 10% speedup): GPU and CPU utility low)
        # nr_gpus = torch.cuda.device_count()
        # exp_utils.print_and_save(self.Config.EXP_PATH, "nr of gpus: {}".format(nr_gpus))
        # self.net = nn.DataParallel(self.net)

        if torch.cuda.device_count() > 1 and self.Config.USE_DP:
            print('Using DataParallel')
            self.net = nn.DataParallel(self.net)
            net = self.net.to("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = self.net.to(self.device)
        
        if self.Config.OPTIMIZER == "AdamW":
            self.optimizer = AdamW(net.parameters(), lr=self.Config.LEARNING_RATE,
                                   weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(net.parameters(), lr=self.Config.LEARNING_RATE,
                                    weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE,
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


    def train(self, X, y, weight_factor=None, timesteps=None):
        X = X.contiguous().cuda(non_blocking=True)  # (bs, features, x, y)
        y = y.contiguous().cuda(non_blocking=True)  # (bs, classes, x, y)

        if self.Config.RESIZE_TO_512:
            X = F.interpolate(X, size=(512, 512), mode='bicubic', align_corners=False) # (bs, features, 144, 144) -> (bs, features, 512, 512)
            y = F.interpolate(y, size=(512, 512), mode='nearest') # (bs, classes, 144, 144) -> (bs, classes, 512, 512)

        if self.Config.MODEL == "MASAM":
            X = X.unsqueeze(0) # (bs(1), z, features, x, y)

        self.net.train()
        self.optimizer.zero_grad() 
        if APEX_AVAILABLE and self.Config.FP16:
            with autocast():
                if self.Config.MODEL == 'LatentDiffusionModel':
                    # Predict noise with the model
                    outputs = self.net(X, timesteps)
                elif self.Config.MODEL == 'MASAM':
                    outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE_TO_512 else 512)
                    outputs = outputs['low_res_logits']
                    # print('outputs.shape:',outputs.shape)
                    #print('outputs:', outputs)
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

        else:        
            outputs = self.net(X)  # (bs, classes, x, y)
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
                else:
                    loss = self.criterion(outputs, y)
        
        if APEX_AVAILABLE and self.Config.FP16:
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def test(self, X, y, weight_factor=None):
        with torch.no_grad():
            X = X.contiguous().cuda(non_blocking=True)
            y = y.contiguous().cuda(non_blocking=True)
        
        if self.Config.RESIZE_TO_512:
            X = F.interpolate(X, size=(512, 512), mode='bicubic', align_corners=False) # (bs, features, 144, 144) -> (bs, features, 512, 512)
            y = F.interpolate(y, size=(512, 512), mode='nearest') # (bs, classes, 144, 144) -> (bs, classes, 512, 512)

        if self.Config.MODEL == "MASAM":
            X = X.unsqueeze(0) # (bs(1), z, features, x, y)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)

        if self.Config.MODEL == 'MASAM':
            outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1] if not self.Config.RESIZE_TO_512 else 512)
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
            weights[bundle_mask.data] *= weight_factor
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
                # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).contiguous().to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)

        if self.Config.MODEL == 'MASAM':
            outputs = self.net(X, multimask_output=True, image_size=self.Config.INPUT_DIM[-1])
            outputs = outputs['low_res_logits']
        else:
            outputs = self.net(X)  # forward
        if self.Config.EXPERIMENT_TYPE == "peak_regression" or self.Config.EXPERIMENT_TYPE == "dm_regression":
            probs = outputs.detach().cpu().numpy()
        else:
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



# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr