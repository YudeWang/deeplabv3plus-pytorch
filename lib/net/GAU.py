# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import numpy as np 

class GAU(nn.Module):
	
	def __init__(self, dim_in, dim_out, resolution_in):
		super(GAU, self).__init__()
		self.conv3x3 = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1)
		self.conv1x1 = nn.Conv2d(dim_out, dim_out, 1, 1, padding=0)
		self.avgpool = nn.AvgPool2d(resolution_in//2, stride=resolution_in//2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

	def forward(self, low, high):
		conv3x3 = self.conv3x3(low)
		gp = self.avgpool(high)
		gp_conv1x1 = self.conv1x1(gp)
		mul = conv3x3 * gp_conv1x1
		high_up = self.upsample(high)
		result = mul + high_up
		
		return result

