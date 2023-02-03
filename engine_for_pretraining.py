# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, momentum_schedule, model_copy,
                    model_copy_without_ddp, temp,  max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    print('The temp is:',temp)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, samples_aug, bool_masked_pos = batch

        bs = samples.shape[0]

        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        samples_aug = samples_aug.to(device, non_blocking=True)

        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        bool_masked_pos_zero = torch.zeros(bool_masked_pos.shape).to(device, non_blocking=True).to(torch.bool)


        length = 196
        labels_idx = torch.arange(length)
        labels_idx_batch = labels_idx.unsqueeze(0).expand(bs, -1)
        labels_idx_batch = labels_idx_batch.to(device, non_blocking=True)
        labels_idx_batch = labels_idx_batch[bool_masked_pos]


        with torch.cuda.amp.autocast():
            with torch.no_grad():
                feat_full = model_copy(samples_aug, bool_masked_pos=bool_masked_pos_zero, return_all_tokens = True, only_return_before_head=True, use_mlp_projectors = True)

        with torch.cuda.amp.autocast():
            feat = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=True, only_return_before_head=True, use_mlp_projectors = True)

            feat_full = nn.functional.normalize(feat_full, p=2, dim=-1)
            feat = nn.functional.normalize(feat, p=2, dim=-1)
            if step == 0:
                print('feat_full',feat_full.shape)
                print('feat', feat.shape)

            logits_ab = torch.matmul(feat, feat_full.permute(0,2,1)) / temp
            logits_ab = logits_ab[bool_masked_pos]

            loss = nn.CrossEntropyLoss()(logits_ab, labels_idx_batch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # EMA update for tokenizer
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in model.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in model_copy_without_ddp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
