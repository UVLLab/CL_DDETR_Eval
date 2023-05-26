#------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
import pickle
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from glob import glob

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    #TODO : clip max grading usually set value 1 or 5 but this therory used to value 0.1 originally
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=3, type=float, # GIOU is Normalized IOU -> False일 때에도, 거리 차이에를 반영할 수 있음(기존의 IOU는 틀린 경우는 전부 0으로써 결과를 예상할 수 없었는데, GIOU는 실제 존재하는 GT BBOX와 Pred BBOX의 거리를 예측하도록 노력하게 됨.)
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    #parser.add_argument('--coco_path', default='/data/LG/real_dataset/total_dataset/didvepz/', type=str)
    # parser.add_argument('--coco_path', default='/home/user/Desktop/vscode/cocodataset/', type=str)
    parser.add_argument('--file_name', default='./saved_rehearsal', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='CL_TEST', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--LG', default=False, action='store_true', help="for LG Dataset process")
    
    #* CL Setting 
    parser.add_argument('--pretrained_model', default="/home/user/Desktop/vscode/CL_DDETR/CCBReplay(COCO_10Epoch_1K_NoFrozen)/cp_02_02_6.pth",
                        type=str, nargs='+', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--start_task', default=0, type=int, metavar='N',help='start task')
    parser.add_argument('--eval', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #* Continual Learning 
    parser.add_argument('--Task', default=2, type=int, help='The task is the number that divides the entire dataset, like a domain.') #if Task is 1, so then you could use it for normal training.
    parser.add_argument('--Task_Epochs', default=3, type=int, help='each Task epoch, e.g. 1 task is 5 of 10 epoch training.. ')
    parser.add_argument('--Total_Classes', default=59, type=int, help='number of classes in custom COCODataset. e.g. COCO : 80 / LG : 59')
    parser.add_argument('--Total_Classes_Names', default=False, action='store_true', help="division of classes through class names (DID, PZ, VE). This option is available for LG Dataset")
    parser.add_argument('--CL_Limited', default=1000, type=int, help='Use Limited Training in CL. If you choose False, you may encounter data imbalance in training.')

    #* Rehearsal method
    parser.add_argument('--Rehearsal', default=False, action='store_true', help="use Rehearsal strategy in diverse CL method")
    parser.add_argument('--Mosaic', default=False, action='store_true', help="use Our CCM strategy in diverse CL method")
    parser.add_argument('--Memory', default=500, type=int, help='memory capacity for rehearsal training')
    parser.add_argument('--Continual_Batch_size', default=4, type=int, help='continual batch training method')
    parser.add_argument('--Rehearsal_file', default='./Rehearsal_dict/', type=str)
    
    #control check version
    parser.add_argument('--save', default=False, action='store_true', help ="save your model output image")
    parser.add_argument('--coco_path', default='/newvetest/10test/', type=str, help="TOTAL : did+ve test (/testdataset/total_test)  | VE : ve test  (../testdataset)")
    parser.add_argument('--all_data', default=False, action='store_true', help ="save your model output image")
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    print(f"current rank : {utils.get_local_rank()}")
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print(f"test directory list : {len(args.pretrained_model)}")
    for enum, predefined_model in enumerate(args.pretrained_model):
        print(f"current predefined_model : {enum}, defined model name : {predefined_model}")
        
        if args.pretrained_model is not None:
            model = load_model_params(model, predefined_model)
        
        model_without_ddp = model
        if args.all_data == True:
            dir_list = glob("/home/user/Desktop/vscode/newvetest/*")
            dir_list.remove("/home/user/Desktop/vscode/newvetest/test_coco.txt")
            # dir_list.remove("/home/user/Desktop/vscode/newvetest/10test")
            # dir_list.remove("/home/user/Desktop/vscode/newvetest/2021")
            # dir_list.remove("/home/user/Desktop/vscode/newvetest/multisingle")
            # dir_list.remove("/home/user/Desktop/vscode/newvetest/")
        else:
            dir_list = ["/home/user/Desktop/vscode"+ args.coco_path]
        #* collate_fn : 최종 출력시에 모든 배치값에 할당해주는 함수를 말함. 여기서는 Nested Tensor 호출을 의미함.

        print(f"check directory list : {dir_list}")
        output_dir = Path(args.output_dir)
        
        print("Start training")
        start_time = time.time()
        file_name = args.file_name + "_" + str(0)
        if args.Task != 1:
            Divided_Classes = DivideTask_for_incre(args.Task, args.Total_Classes, args.Total_Classes_Names)
            if args.Total_Classes_Names == True :
                args.Task = len(Divided_Classes)
        DIR = './mAP_TEST.txt'
        filename_list = ["didtest", "pztest", "VE2021", "VEmultisingle", "VE10test"]
        if args.eval:
            for task_idx, cur_file_name in enumerate(filename_list):
                cur_file_name = filename_list[task_idx]
                    
                file_link = [name for name in dir_list if cur_file_name == os.path.basename(name)]
                args.coco_path = file_link[0]
                print(f"file name : {args.coco_path}")
                if 'VE' in cur_file_name:
                    task_idx = 2
                    
                print(f"trained task classes: {Divided_Classes[task_idx]}")
                dataset_val, data_loader_val, _, _  = Incre_Dataset(task_idx, args, Divided_Classes[task_idx], False)
                base_ds = get_coco_api_from_dataset(dataset_val)
                
                
                with open(DIR, 'a') as f:
                    f.write(f"----------------------------------------------------------------\n")
                    f.write(f"NOW TASK num : {task_idx}, checked classes : {Divided_Classes[task_idx]} \t file_name : {str(predefined_model)} \n")
                _, _ = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, DIR)
            continue

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
