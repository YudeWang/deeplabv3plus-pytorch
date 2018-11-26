# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.sync_batchnorm import SynchronizedBatchNorm2d
from net.backbone import build_backbone
from net.ASPP import ASPP

class SuperNet(nn.Module):
	def __init__(self, cfg):
		super(SuperNet, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				resolution_in=cfg.MODEL_ASPP_RESOLUTION)
		self.merge1 = merge_block(cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_NUM_CLASSES,cfg.MODEL_ASPP_OUTDIM,cfg.DATA_RESCALE,scale=4)
		self.merge4 = merge_block(cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_NUM_CLASSES,cfg.MODEL_ASPP_OUTDIM,cfg.DATA_RESCALE//4,scale=2)
		self.merge8 = merge_block(cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_NUM_CLASSES,cfg.MODEL_ASPP_OUTDIM,cfg.DATA_RESCALE//8,scale=2)
		
		self.cut1 = shortcut_block(3, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, dilation=1) 
		self.cut4 = shortcut_block(256, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, dilation=1) 
		self.cut8 = shortcut_block(512, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, dilation=1) 

		self.diff = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=2, dilation=2),
				nn.ReLU(inplace=True),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = build_backbone(cfg.MODEL_BACKBONE)		
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		f16 = self.aspp(layers[-1])
		f16_cls = self.cls_conv(f16)
		f16_sig = torch.sigmoid(f16_cls)
			
		f8_cut = self.cut8(layers[1])		
		f8 = self.merge8(f8_cut, f16, f16_cls)
		f8_cls = self.cls_conv(f8)
		f8_sig = torch.sigmoid(f8_cls)

		f4_cut = self.cut4(layers[0])
		f4 = self.merge4(f4_cut, f8, f8_cls)
		f4_cls = self.cls_conv(f4)
		f4_sig = torch.sigmoid(f4_cls)
		
		f1_cut = self.cut1(x)
		f1 = self.merge1(f1_cut, f4, f4_cls)
		f1_cls = self.cls_conv(f1)

		return f1_cls, f4_sig, f8_sig, f16_sig

class shortcut_block(nn.Module):
	
	def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation):
		super(shortcut_block, self).__init__()
		self.block = nn.Sequential(
				nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding=padding, dilation=dilation),
				nn.ReLU(inplace=True),
		)
	
	def forward(self,x):
		return self.block(x)

class merge_block(nn.Module):
	
	def __init__(self, shortcut_channel, up_channel, cls_channel, output_channel, resolution, scale=2):
		super(merge_block, self).__init__()
		self.cat_conv = nn.Sequential(
				nn.Conv2d(shortcut_channel + up_channel, output_channel, 3, 1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(output_channel, output_channel, 3, 1, padding=1),
				nn.ReLU(inplace=True),
				)
		self.diff = nn.Sequential(
				nn.Conv2d(cls_channel, cls_channel, 3, 1, padding=2, dilation=2),
				nn.ReLU(inplace=True)
				)
		self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
		self.resolution = resolution
	
	def forward(self, cut, fm, cls):
		diff = self.diff(cls)
		diff_up = self.upsample(diff)
		fm_up = self.upsample(fm)
		diff_att = torch.sum(diff_up, dim=1).view(-1, 1, self.resolution, self.resolution)
		cut_att = cut * diff_att
		cat = torch.cat([fm_up, cut], 1)
		result = self.cat_conv(cat)
		return result
