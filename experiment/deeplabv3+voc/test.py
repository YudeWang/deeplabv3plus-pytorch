# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim


from torch.utils.data import DataLoader

def test_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()	
	result_list = []
	 
	for i_batch, sample_batched in enumerate(dataloader):
	
		inputs_batched = sample_batched['image']
		name_batched = sample_batched['name']
		row_batched = sample_batched['row']
		col_batched = sample_batched['col']
		predicts_batched = net(inputs_batched)
		
		if cfg.TEST_FLIP:
			inputs_batched_flip = torch.flip(inputs_batched,[3]) 
			predicts_batched_flip = torch.flip(net(inputs_batched_flip),[3])
			predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0

		predicts_batched = torch.argmax(predicts_batched, dim=1).cpu().numpy().astype(np.uint8)
		for i in range(inputs_batched.shape[0]):
			row = row_batched[i]
			col = col_batched[i]
			# max_edge = np.max(row,col)
			# rate = self.DATA_RESCALE / max_edge
			# new_row = row*rate
			# new_col = col*rate
			# s_row = (self.DATA_RESCALE-new_row)//2
			# s_col = (self.DATA_RESCALE-new_col)//2
 
			# p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
			# p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
			p = cv2.resize(predicts_batched[i,:,:], dsize=(col,row), interpolation=cv2.INTER_NEAREST)
			result_list.append({'predict':p, 'name':name_batched[i]})

		print('%d/%d'%(i_batch,len(dataloader)))
	dataset.save_result(result_list, cfg.MODEL_NAME)
	dataset.do_matlab_eval(cfg.MODEL_NAME)
	print('Test finished')

if __name__ == '__main__':
	test_net()


