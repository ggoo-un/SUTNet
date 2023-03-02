import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_g, define_d
from basicsr.models.gan_model import GanModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanGanModel(GanModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanGanModel, self).__init__(opt)

        # self.scaler = torch.cuda.amp.GradScaler()

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.is_train:
            self.net_d = define_d(deepcopy(opt['network_d']))
            self.net_d = self.model_to_device(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        train_opt = self.opt['train']

        # load
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_g(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path_g = self.opt['path'].get('pretrain_network_g', None)
            if load_path_g is not None:
                self.load_network(self.net_g_ema, load_path_g,
                                  self.opt['path'].get('strict_load_g',True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            if load_path_d is not None:
                self.load_network(self.net_d, load_path_d,
                                  self.opt['path'].get('strict_load_d',True), 'params_ema')

        # define losses
        # ------------------------------------
        # G_loss
        # ------------------------------------
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type') # L1loss
            G_lossfn_cls = getattr(loss_module, pixel_type)
            self.G_lossfn = G_lossfn_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('net g pixel loss are None.')

        # ------------------------------------
        # D_loss
        # ------------------------------------
        if train_opt.get('d_opt'):
            d_loss_type = train_opt['d_opt'].pop('type') # GANLoss
            D_lossfn_cls = getattr(loss_module, d_loss_type)
            self.D_lossfn = D_lossfn_cls(**train_opt['d_opt']).to(self.device)
        else:
            raise ValueError('net d loss are None.')
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), **train_opt['optim_d'])
        elif optim_type == 'AdamW':
            self.optimizer_d = torch.optim.AdamW(self.net_d.parameters(), **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        # with torch.cuda.amp.autocast():
        # net_g forward
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1].contiguous()

        loss_dict = OrderedDict()
        # net_g pixel loss
        loss_G_total = 0.
        for pred in preds:
            loss_G_total += self.G_lossfn(pred, self.gt)

        #######################################################################

        if self.opt['train']['d_opt']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            pred_g_fake = self.net_d(self.output)
            D_loss = self.D_lossfn(pred_g_fake, True)
        elif self.opt['train']['d_opt']['gan_type'] == 'ragan':
            pred_d_real = self.net_d(self.gt).detach()
            pred_g_fake = self.net_d(self.output)
            D_loss = (self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                    self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
        loss_G_total += D_loss  # 3) GAN loss

            #######################################################################
        loss_dict['loss_G'] = loss_G_total
        loss_G_total.backward()
        # self.scaler.scale(loss_G_total.contiguous()).backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        # self.scaler.step(self.optimizer_g)
        # self.scaler.update()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # with torch.cuda.amp.autocast():
        loss_D_total = 0
        pred_d_fake = self.net_d(self.output.detach())  # 1) fake data, detach to avoid BP to G
        pred_d_real = self.net_d(self.gt)  # 2) real data
        if self.opt['train']['d_opt']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            l_d_real = self.D_lossfn(pred_d_real, True)
            l_d_fake = self.D_lossfn(pred_d_fake, False)
            loss_D_total = l_d_real + l_d_fake
        elif self.opt['train']['d_opt']['gan_type'] == 'ragan':
            l_d_real = self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
            l_d_fake = self.D_lossfn(pred_d_fake - torch.mean(pred_d_real, 0, True), False)
            loss_D_total = (l_d_real + l_d_fake) / 2

        loss_dict['loss_D'] = loss_D_total
        # self.scaler.scale(loss_D_total).backward()
        # self.scaler.step(self.optimizer_d)
        # self.scaler.update()
        loss_D_total.backward()
        self.optimizer_d.step()

        # ------------------------------------
        # record log
        # ------------------------------------
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()  # padding

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            ## 측정
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
