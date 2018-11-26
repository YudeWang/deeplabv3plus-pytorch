# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import json
import torch
from torch.utils.data import Dataset
import cv2
from scipy.misc import imread
import numpy as np
from datasets.transform import *
from datasets.metric import AverageMeter, accuracy, intersectionAndUnion

class ADE20KDataset(Dataset):
    def __init__(self, dataset_name, cfg, period):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data')
        self.rst_dir = os.path.join(self.root_dir,'ADEChallengeData2016','result')
        self.period = period
        self.cfg = cfg
        self.num_categories = 150
        assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.totensor = ToTensor()

        self.odgt = None        

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale((cfg.DATA_RESCALE, cfg.DATA_RESCALE))

        if self.period == 'train':
            self.odgt = os.path.join(self.root_dir,'ADEChallengeData2016','train.odgt')
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.odgt = os.path.join(self.root_dir,'ADEChallengeData2016','validation.odgt')

        self.list_sample = [json.loads(x.rstrip()) for x in open(self.odgt, 'r')]

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.list_sample[idx]['fpath_img'])
        img = imread(image_path, mode='RGB')
        assert(img.ndim == 3)
        r = self.list_sample[idx]['height']
        c = self.list_sample[idx]['width']

        name = self.list_sample[idx]['fpath_img'].replace('ADEChallengeData2016/images/','')
        if self.period == 'train':
            name = name.replace('train/','') 
        if 'val' in self.period:
            name = name.replace('validation/','') 
        assert(self.period != 'test')
        name = name.replace('.jpg','')
        
        sample = {'image': img, 'name': name, 'row': r, 'col': c}

        if self.period == 'train':
            seg_path = os.path.join(self.root_dir, self.list_sample[idx]['fpath_segm'])
            seg = imread(seg_path)
            #seg[seg>=self.cfg.MODEL_NUM_CLASSES] = 0
            #seg += 1
            sample['segmentation'] = seg
            assert(seg.ndim == 2)
            assert(img.shape[0] == seg.shape[0])
            assert(img.shape[1] == seg.shape[1])

            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)

        if self.cfg.DATA_RESCALE > 0:
            sample = self.rescale(sample)
        if 'segmentation' in sample.keys():
            sample['segmentation_onehot'] = onehot(sample['segmentation'], self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample
 
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap

    def save_result(self, result_list, model_id):
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s'%model_id)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path,'%s.png'%sample['name'])
            '''

            ATTENTION!!!

            predict label start from 0 or -1 ?????

            DO NOT have operation here!!!


            '''
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def do_python_eval(self, model_id):
        folder_path = os.path.join(self.rst_dir,'%s'%model_id)

        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        for sample in self.list_sample:
            name = sample['fpath_img'].replace('ADEChallengeData2016/images/','')
            if self.period == 'train':
                name = name.replace('train/','') 
            if 'val' in self.period:
                name = name.replace('validation/','') 
            assert(self.period != 'test')
            name = name.replace('.jpg','')

            predict_path = os.path.join(folder_path,'%s.png'%name)
            label_path = os.path.join(self.root_dir, sample['fpath_segm'])
            
            predict = imread(predict_path)
            label = imread(label_path)

            acc, pix = accuracy(predict, label)
            intersection, union = intersectionAndUnion(predict, label, self.num_categories)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            print('class [{}], IoU: {}'.format(i, _iou))

        print('[Eval Summary]:')
        print('Mean IoU: {:.4}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average()*100))

