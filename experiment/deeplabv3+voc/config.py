# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = '/home/wangyude/project/segmentation'

		self.DATA_NAME = 'VOC2012'
		self.DATA_WORKERS = 4
		self.DATA_RESCALE = 512
		self.DATA_RANDOMCROP = 512
		self.DATA_RANDOMROTATION = 0
		self.DATA_RANDOMSCALE = 2
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0.5
		
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_RESOLUTION = None
		self.MODEL_ASPP_RESOLUTION = 32
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_NUM_CLASSES = 21
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model')

		self.TRAIN_LR = 0.01
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_POWER = 0.9
		self.TRAIN_GPUS = 2
		self.TRAIN_BATCHES = 16
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 60
		self.TRAIN_LR_MST = [100,140]
		self.TRAIN_TBLOG = True
		self.TRAIN_CKPT = None#'/home/wangyude/project/segmentation/model/deeplabv3plus_res101_atrous_COCO2017_all.pth'

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log')

		self.TEST_FLIP = False
		self.TEST_CKPT = '/home/wangyude/project/segmentation/model/deeplabv3plus_res101_atrous_VOC2012_itr30000.pth'
		self.TEST_GPUS = 2
		self.TEST_BATCHES = 12		

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		for i in range(len(self.TRAIN_LR_MST)):
			self.TRAIN_LR_MST[i] = self.TRAIN_LR_MST[i] - self.TRAIN_MINEPOCH
	
	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)



cfg = Configuration() 	
