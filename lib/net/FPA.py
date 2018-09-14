# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import numpy as np 

class FPA(nn.Module):
	
	def __init__(self, dim_in, dim_out, resolution_in):
		super(FPA, self).__init__()
		self.conv1x1 = nn.Conv2d(dim_in,dim_out, 1, 1, padding=0)
		self.conv7x7_1 = nn.Conv2d(dim_in, dim_in, 7, 2, padding=3)
		self.conv7x7_2 = nn.Conv2d(dim_in, dim_out, 7, 1, padding=3)
		self.conv5x5_1 = nn.Conv2d(dim_in, dim_in, 5, 2, padding=2)
		self.conv5x5_2 = nn.Conv2d(dim_in, dim_out, 5, 1, padding=2)
		self.conv3x3_1 = nn.Conv2d(dim_in, dim_in, 3, 2, padding=1)
		self.conv3x3_2 = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1)
		self.conv1x1_gp = nn.Conv2d(dim_in, dim_out, 1, 1, padding=0)

		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
		self.upsample_gp = nn.Upsample(scale_factor=resolution_in, mode='nearest')
	
		self.avgpool = nn.AvgPool2d(resolution_in, stride=resolution_in)

	def forward(self, x):
		conv1x1 = self.conv1x1(x)
		
		conv7x7_1 = self.conv7x7_1(x)
		conv7x7_2 = self.conv7x7_2(conv7x7_1)

		conv5x5_1 = self.conv5x5_1(conv7x7_1)
		conv5x5_2 = self.conv5x5_2(conv5x5_1)

		conv3x3_1 = self.conv3x3_1(conv5x5_1)
		conv3x3_2 = self.conv3x3_2(conv3x3_1)

		conv3up = self.upsample(conv3x3_2)
		conv5up = self.upsample(conv3up + conv5x5_2)
		conv7up = self.upsample(conv5up + conv7x7_2)
		
		conv_multiply = conv7up * conv1x1

		avgpool = self.avgpool(x)
		avgconv = self.conv1x1_gp(avgpool)
		avgup = self.upsample_gp(avgconv)

		result = conv_multiply + avgup		
		return result

