# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import numpy as np
from typing import Iterable
from tqdm import tqdm
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher
import os
import time
from typing import Tuple, Collection, Dict, List
from datasets import build_dataset, get_coco_api_from_dataset
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from custom_training import *
from custom_prints import check_components

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)


def train_one_epoch(args, epo, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, MosaicBatch: Boolean,  
                    current_classes: List = [], rehearsal_classes: Dict = {}):
    ex_device = torch.device("cpu")
    if MosaicBatch == False:    
        prefetcher = data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True, Mosaic=True)


    set_tm = time.time()
    sum_loss = 0.0
    count = 0
    label_dict = {} #* 하나의 에포크에서 계속해서 Class Check을 위한 딕셔너리 생성
    early_stopping_count = 0
    for idx in tqdm(range(len(data_loader))): #targets 
        with torch.no_grad():
            torch.cuda.empty_cache()
            samples, targets, origin_samples, origin_targets = prefetcher.next()
            #print(f"target value: {targets}")
            if idx > 100000:
                break
        
            if early_stopping_count > 40 :
                dist.barrier()
                print(f"too many stopping index.")
                break
            
            train_check = True
            samples = samples.to(ex_device)
            targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        
            #TODO : one samples no over / one samples over solve this ! 
            
            #* because MosaicAugmentation Data has not original data
            no_use, yes_use, label_dict = check_class(args.verbose, args.LG , targets, label_dict, current_classes, CL_Limited=args.CL_Limited) #! Original에 한해서만 Limited Training(현재 Task의 데이터에 대해서만 가정)
            samples, targets, origin_samples, origin_targets, train_check = decompose_dataset(no_use_count=len(no_use), samples= samples, targets = targets, origin_samples=origin_samples, origin_targets= origin_targets ,used_number= yes_use)
            trainable = check_training_gpu(train_check=train_check)
            if trainable == False :
                del samples, targets, origin_samples, origin_targets, train_check
                torch.cuda.empty_cache()
                early_stopping_count += 1
                if MosaicBatch == True :
                    _, _, _, _ = prefetcher.next(new = True)
                    
                continue
                
    
        if trainable == True:
        #contruct rehearsal buffer in main training
            rehearsal_classes, sum_loss, count = Original_training(args, epo, idx, count, sum_loss, samples, targets, origin_samples, origin_targets, 
                                                model, criterion, optimizer, rehearsal_classes, train_check, current_classes)

        early_stopping_count = 0
        #* For Mosaic Training method
        if MosaicBatch == True and trainable == True:
            samples, targets, _, _ = prefetcher.next() #* Different
            count, sum_loss = Mosaic_training(args, epo, idx, count, sum_loss, samples, targets, model, criterion, optimizer, current_classes, "currentmosaic")
            
            samples, targets, _, _ = prefetcher.next() #* Next samples
            count, sum_loss = Mosaic_training(args, epo, idx, count, sum_loss, samples, targets, model, criterion, optimizer, current_classes, "differentmosaic")
            
        del samples, targets, trainable, train_check
        torch.cuda.empty_cache()
        
    #for checking limited Classes Learning
    check_components("Limited", label_dict, args.verbose)
    if utils.is_main_process():
        print("Total Time : ", time.time() - set_tm)
    return rehearsal_classes

@torch.no_grad()
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        
        # 여기서 적힌 samples는 전부 Nested Tensor 형태
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model, teacher_model, criterion, postprocessors, data_loader, base_ds, device, output_dir, DIR) :
    model.eval()
    criterion.eval()
    teacher_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, DIR)

        
    for samples, targets, _, _ in metric_logger.log_every(data_loader, 10, header):
        # t_encoder_output_weight = []
        # s_encoder_output_weight = []
        # def t_hook_fn(module, input, output):
        #     t_encoder_output_weight.append(output)
            
        # def s_hook_fn(module, input, output):
        #     s_encoder_output_weight.append(output)
            
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # hook = teacher_model.transformer.encoder.register_forward_hook(t_hook_fn)
        # hook_studnet = model.transformer.encoder.register_forward_hook(s_hook_fn)
        
        with torch.no_grad():
            teacher_model(samples)
            
        outputs = model(samples)
        # hook.remove()
        # hook_studnet.remove()
        # #encoder_output = model.transformer.memory
        # distill_loss = torch.nn.functional.smooth_l1_loss(s_encoder_output_weight[0], t_encoder_output_weight[0].detach())
        # distill_loss /= samples.shape[0]
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict, True)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #print(results)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

