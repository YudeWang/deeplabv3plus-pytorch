# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform_matting import *

class MattingDataset(Dataset):
    def __init__(self, dataset_name='matting', cfg, period):
        super(MattingDataset,self).__init__()
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data','Matting')
        self.dataset_dir = self.root_dir
        self.rst_dir = os.path.join(self.root_dir,'results')
        self.period = period
        file_name = self.dataset_dir+'/'+period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.totensor = ToTensor()
        self.cfg = cfg

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale((cfg.DATA_RESCALE,cfg.DATA_RESCALE),is_continuous=True)
        if self.period == 'train':        
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION,is_continuous=True)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE,is_continuous=True)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.dataset_dir + '/' + self.period + '/' + name + '.png'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        
        if self.period == 'train':
            seg_file = self.dataset_dir + '/' + self.period + '/' + name + '_matte.png'
            segmentation = cv2.imread(seg_file)
            sample['segmentation'] = segmentation

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
        sample = self.totensor(sample)

        return sample

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s.png'%sample['name'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def do_python_eval(self, model_id):
        result_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        label_path = os.path.join(self.dataset_dir,self.period)
        sum_error = 0
        count = 0
        for name in self.name_list:
            predict_file = os.path.join(result_path, '%s.png'%name)
            gt_file = os.path.join(label_path, '%s_matte.png'%name)
            predict = cv2.imread(predict_file)
            gt = cv2.imread(gt_file)
            r,c = gt.shape
            error = np.sum(((gt-predict)/255)**2)
            sum_erro += error
            count += 1
        sum_error /= count
        print('mean square error: %f'%sum_error)
