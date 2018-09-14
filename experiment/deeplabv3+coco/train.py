# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def train_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)
	
	net = generate_net(cfg)
	if cfg.TRAIN_TBLOG:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)

	

	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	print('module to device')
	net.to(device)		
	print('module parallel')
	if cfg.TRAIN_GPUS > 1:
		net = nn.DataParallel(net)
	print('load pretrained parameters')
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT))
	print('initialize others')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM)
#	optimizer = nn.DataParallel(optimizer)
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	running_loss = 0.0
	tblogger = SummaryWriter(cfg.LOG_DIR)

	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		# scheduler.step()
		# now_lr = scheduler.get_lr()
		for i_batch, sample_batched in enumerate(dataloader):
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			labels_batched = labels_batched.long().to(1)
			optimizer.zero_grad()
			#inputs_batched.to(0)
			predicts_batched = net(inputs_batched).to(1)
			loss = criterion(predicts_batched, labels_batched)

			loss.backward()
			now_lr = adjust_lr(optimizer, itr, max_itr)
			optimizer.step()

			running_loss += loss.item()
			
			print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g ' % 
				(epoch, cfg.TRAIN_EPOCHS, i_batch, dataset.__len__()//cfg.TRAIN_BATCHES,
				itr+1, now_lr, running_loss))
			if cfg.TRAIN_TBLOG and itr%100 == 0:
				inputs = inputs_batched.numpy()[0]/2.0 + 0.5
				labels = labels_batched[0].cpu().numpy()
				labels_color = dataset.label2colormap(labels).transpose((2,0,1))
				predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
				predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))				

				tblogger.add_scalar('loss', running_loss, itr)
				tblogger.add_scalar('lr', now_lr, itr)
				tblogger.add_image('Input', inputs, itr)
				tblogger.add_image('Label', labels_color, itr)
				tblogger.add_image('Output', predicts_color, itr)
			running_loss = 0.0
			
			if itr % 5000 == 4999:
				save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,itr+1))
				torch.save(net.state_dict(), save_path)
				print('%s has been saved'%save_path)

			itr += 1
		
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME))		
	torch.save(net.state_dict(),save_path)
	if cfg.TRAIN_TBLOG:
		tblogger.close()
	print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
	for group in optimizer.param_groups:
		group['lr'] = now_lr
	return now_lr

if __name__ == '__main__':
	train_net()


