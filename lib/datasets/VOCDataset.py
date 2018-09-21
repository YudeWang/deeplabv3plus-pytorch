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
from datasets.transform import *

class VOCDataset(Dataset):
    def __init__(self, dataset_name, cfg, period):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit')
        self.dataset_dir = os.path.join(self.root_dir,dataset_name)
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.period = period
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.dataset_dir, 'Annotations')
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
        self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation')
        file_name = self.set_dir+'/'+period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomhsv = None
        self.totensor = ToTensor()
        self.cfg = cfg
	
        if dataset_name == 'VOC2012':
            self.categories = [
                        'aeroplane',    #1
                        'bicycle',      #2
                        'bird',         #3
                        'boat',         #4
                        'bottle',       #5
                        'bus',          #6
                        'car',          #7
                        'cat',          #8
                        'chair',        #9
                        'cow',          #10
                        'diningtable',  #11
                        'dog',          #12
                        'horse',        #13
                        'motorbike',    #14
                        'person',       #15
                        'pottedplant',  #16
                        'sheep',        #17
                        'sofa',         #18
                        'train',        #19
                        'tvmonitor']    #20
            self.coco2voc = [[0],
                             [5],
                             [2],
                             [16],
                             [9],
                             [44],#,46,86],
                             [6],
                             [3,8],
                             [17],
                             [62],
                             [21],
                             [67],
                             [18],
                             [19],#,24],
                             [4],
                             [1],
                             [64],
                             [20],
                             [63],
                             [7],
                             [72]]

            self.num_categories = len(self.categories)
            #assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale((cfg.DATA_RESCALE,cfg.DATA_RESCALE))
        if self.period == 'train':        
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0 or cfg.DATA_RANDOMSCALE != 1:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION,cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name + '.jpg'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        
        if self.period == 'train':
            seg_file = self.seg_dir + '/' + name + '.png'
            segmentation = np.array(Image.open(seg_file))
            segmentation[segmentation>len(self.categories)] = 0
            sample['segmentation'] = segmentation

            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0 or self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)

        if self.cfg.DATA_RESCALE > 0:
            sample = self.rescale(sample)
        if 'segmentation' in sample.keys():
            sample['segmentation_onehot'] = onehot(sample['segmentation'], self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample



    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap
    
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap
    
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
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def do_matlab_eval(self, model_id):
        import subprocess
        path = os.path.join(self.root_dir, 'VOCcode')
        eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; VOCinit; '
        cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
        cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
        cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
        cmd += 'quit;"'

        print('start subprocess for matlab evaluation...')
        print(cmd)
        subprocess.call(cmd, shell=True)

    def __coco2voc(self, m):
        r,c = m.shape
        result = np.zeros((r,c),dtype=np.uint8)
        for i in range(0,21):
            for j in self.coco2voc[i]:
                result[m==j] = i
        return result
