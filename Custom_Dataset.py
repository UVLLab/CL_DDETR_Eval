from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
import random
import albumentations as A
from util.box_ops import box_cxcywh_to_xyxy_resize, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from Custom_augmentation import CCB
from torch.utils.data.sampler import SubsetRandomSampler
def Incre_Dataset(Task_Num, args, Incre_Classes, Train = True):    

    current_classes = Incre_Classes
    print(f"current_classes : {current_classes}")
    
    if Train :
        if len(Incre_Classes) == 1:
            dataset_train = build_dataset(image_set='train', args=args, class_ids=None) #* Task ID에 해당하는 Class들만 Dataset을 통해서 불러옴
        else: 
            if Task_Num == 0 : #* First Task training
                dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
            else:
                dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
    dataset_val = build_dataset(image_set='val', args=args, class_ids=current_classes)
    
    if args.distributed:
        if args.cache_mode:
            # sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            # sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=True)
    else:
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    # print(f"dataset config :{dataset_train}")
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                             collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                             pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    if Train is True:
        return dataset_train, data_loader_train, sampler_train, current_classes
    else:
        return dataset_val, data_loader_val, sampler_val, current_classes


def DivideTask_for_incre(Task_Counts: int, Total_Classes: int, DivisionOfNames: Boolean):
    '''
        DivisionofNames == True인 경우 Task_Counts는 필요 없어짐 Domain을 기준으로 class task가 자동 분할
        False라면 Task_Counts, Total_Classes를 사용해서 적절하게 분할
        #Task : 테스크의 수
        #Total Class : 총 클래스의 수
        #DivisionOfNames : Domain을 사용해서 분할
    '''
    if DivisionOfNames is True:
        Divided_Classes = []
        Divided_Classes.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]) #DID
        Divided_Classes.append([28, 32, 35, 41, 56]) #PZ , 
        Divided_Classes.append([24, 27, 36, 42, 43, 48, 52]) #VE specific , 
        #Divided_Classes.append([22, 23, 24, 25, 26, 27, 29, 30, 31, 33,34,36, 37, 38, 39, 40,42,43,44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59]) #VE
        return Divided_Classes
    
    classes = [idx+1 for idx in range(Total_Classes)]
    Task = int(Total_Classes / Task_Counts)
    Rest_Classes_num = Total_Classes % Task_Counts
    
    start = 0
    end = Task
    Divided_Classes = []
    for _ in range(Task_Counts):
        Divided_Classes.append(classes[start:end])
        start += Task
        end += Task
    if Rest_Classes_num != 0:
        Rest_Classes = classes[-Rest_Classes_num:]
        Divided_Classes[-1].extend(Rest_Classes)
    
    return Divided_Classes

#현재 (Samples, Targets)의 정보를 가진 형태로 데이터가 구성되어 있음(딕셔너리로 각각의 Class 정보를 가진 채로 구성됨)
#참고로 Samples -> NestedTensor, Target -> List 형태로 구성되어 있음 다만 1개의 
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = re_dict
        self.keys = list(self.re_dict.keys()) #image_id
        self.old_classes = old_classes
        self.datasets = build_dataset(image_set='val', args=args, class_ids=self.old_classes, img_ids=self.keys)
    
    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets


    
class BatchMosaicAug(torch.utils.data.Dataset):
    def __init__(self, datasets, CCB_augmentation, old_length, Mosaic=False ):
        self.Datasets = datasets
        self.Confidence = 0
        self.Mosaic = Mosaic
        self.img_size = (960, 1280) #Height, Width
        self.old_length = old_length
        if self.Mosaic == True: 
            #self._CCB = CCB_augmentation(self.Datasets,  self.Rehearsal_dataset, self.Current_dataset, self.img_size)
            self._CCB = CCB_augmentation(self.img_size)
        
    def __len__(self):
            return len(self.Datasets)    
        
    def __getitem__(self, index):
        img, target, origin_img, origin_target = self.Datasets[index]

        if self.Mosaic == True :
            Current_mosaic_index = self._Mosaic_index(index,)
            image_list = []
            target_list = []
            for index in Current_mosaic_index:
                _, _ , o_img, otarget = self.Datasets[index]
                image_list.append(o_img)
                target_list.append(otarget)
                
            Cur_img, Cur_lab, Dif_img, Dif_lab = self._CCB(image_list, target_list)
                        
            return img, target, origin_img, origin_target, Cur_img, Cur_lab, Dif_img, Dif_lab
        else:
            return img, target, origin_img, origin_target
    
    def _Mosaic_index(self, index): #* Done
        '''
            Only Mosaic index printed 
            index : index in dataset (total dataset = old + new )
            #TODO : count class variables need !! 
        '''
        #self.current_id.add(index)
        #*Curretn Class augmentation / Other class AUgmentation
        #Mosaic_index = random.sample(range(len(self.Current_dataset)), 3)
        #Mosaic_index = random.sample(range(len(self.old)), 3)
        Rehearsal_index = random.sample(range(self.old_length), 3)
            
        #Mosaic_index.insert(0, index)
        Rehearsal_index.insert(0, index)

        return random.sample(Rehearsal_index, len(Rehearsal_index))
    
#For Rehearsal
def CombineDataset(args, RehearsalData, CurrentDataset, Worker, Batch_size, old_classes):
    OldDataset = CustomDataset(args, RehearsalData, old_classes) #oldDatset[idx]:
    old_length = len(OldDataset)
    CombinedDataset = ConcatDataset([OldDataset, CurrentDataset]) #Old : previous, Current : Now
    MosaicBatchDataset = BatchMosaicAug(CombinedDataset, CCB, old_length, args.Mosaic) #* if Mosaic == True -> 1 batch(divided three batch/ False -> 3 batch (only original)
    print(f"current Dataset length : {len(CurrentDataset)} -> Rehearsal + Current length : {len(MosaicBatchDataset)}")
    print(f"********** sucess combined Dataset ***********")
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(MosaicBatchDataset)
        else:
            sampler_train = samplers.DistributedSampler(MosaicBatchDataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(MosaicBatchDataset, shuffle=True)
    
    def worker_init_fn(worker_id):
        torch.manual_seed(worker_id)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(MosaicBatchDataset, batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn, num_workers=Worker,
                        pin_memory=True, prefetch_factor=4, worker_init_fn=worker_init_fn, persistent_workers=args.Mosaic)
    
    
    return MosaicBatchDataset, CombinedLoader, sampler_train