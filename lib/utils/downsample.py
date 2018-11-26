#=========================================
# Written by Yude Wang
#=========================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def max_downsample(m, rate):
	assert(len(m.shape) == 4)
	batch,channel,row,col = m.shape
	min_num = 0
	l = row%rate
	pad = np.ones((batch,channel,l,col),dtype=m.dtype)*min_num
	m_pad = np.concatenate([m,pad],axis=2)
	
	batch,channel,row,col = m_pad.shape
	l = col%rate
	pad = np.ones((batch,channel,row,l),dtype=m_pad.dtype)*min_num
	m_pad = np.concatenate([m_pad,pad],axis=3)
	result = np.amax([m_pad[:,:,(i//rate)::rate, (i%rate)::rate] for i in range(rate*rate)],axis=0)
	return result

def pyramid_label(label, rate, flag, resolution):
	batch,channel,row,col = label.shape
	result_list = []
	weight_list = []
	l = label
	for i in rate:
		n = None
		if flag == 'max':
			n = F.max_pool2d(l, kernel_size=i, stride=i)
		elif flag == 'avg':
			n = F.avg_pool2d(l, kernel_size=i, stride=i)
		else:
			raise ValueError('downsample.py: flag in pyramid_label() is not support yet')
		result_list.append(n)

	for i in range(0, len(result_list)):
		area = (resolution//rate[i]) ** 2
		l = result_list[i]
		w = torch.sum(l, (2,3)).view(-1,channel,1,1)
		w = ((area-w)/area+1)/area/channel
		weight_list.append(w)

	return result_list, weight_list
