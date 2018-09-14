# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP
class deeplabv3plus(nn.Module):
	def __init__(self, cfg):
		super(deeplabv3plus, self).__init__()
		self.backbone = build_backbone(cfg.MODEL_BACKBONE)		
		self.backbone_layers = self.backbone.get_layers()
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				resolution_in=cfg.MODEL_ASPP_RESOLUTION)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(256, cfg.MODEL_SHORTCUT_DIM, 1, 1, padding=0),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),				
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0),
		)
		
#		for m in self.modules():
#			if isinstance(m, nn.Conv2d):
#				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#			elif isinstance(m, nn.BatchNorm2d):
#				nn.init.constant_(m.weight, 1)
#				nn.init.constant_(m.bias, 0)


	def forward(self, x):
		x = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.upsample(feature_aspp)
		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat)
		result = self.upsample(result)	
		return result
