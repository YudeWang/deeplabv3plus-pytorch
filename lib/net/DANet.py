# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as visionF
import PIL
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP

class DANet(nn.Module):
	def __init__(self, cfg):
		super(DANet, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=16//cfg.MODEL_OUTPUT_STRIDE)
#		self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)
		self.downsample4 = nn.AvgPool2d(4,4,0)
		self.downsample_sub = nn.AvgPool2d(cfg.MODEL_OUTPUT_STRIDE//4,cfg.MODEL_OUTPUT_STRIDE//4,0)

		indim = 256
		self.shortcut_conv = shortcut_block(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2, dilation=1)
#		self.shortcut_conv = nn.Sequential(
#				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, cfg.MODEL_SHORTCUT_KERNEL//2, padding=1),
#				nn.ReLU(inplace=True),		
#		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),
		#		nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0),				
		)
		self.sigmoid = nn.Sigmoid()
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)

#		for m in self.modules():
#			if isinstance(m, nn.Conv2d):
#				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#			elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
#				nn.init.constant_(m.weight, 1)
#				nn.init.constant_(m.bias, 0)

		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)		
		self.backbone_layers = self.backbone.get_layers()
		self.cfg = cfg

	def forward(self, x):
		[b,c,row,col] = x.size()
		bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
#		feature_aspp = self.dropout(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		x4 = self.downsample4(x)
		x_bottom = self.downsample_sub(x4)
		x4_up = self.upsample_sub(x_bottom)
		x_up = self.upsample4(x4_up)
		#feature_shallow = self.shortcut_conv(down_x)
		delta4 = x4-x4_up
		delta4 = torch.sum(delta4,dim=1).view(-1,1,row//4,col//4)
		delta1 = torch.abs(x-x_up)
		delta1 = torch.sum(delta1,dim=1).view(-1,row,col)/3
		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)*delta4
		result = self.cat_conv(feature_cat)
		result = self.cls_conv(result+feature_aspp)
		result = self.upsample4(result)
		return result, delta1

class shortcut_block(nn.Module):        
        def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation):
                super(shortcut_block, self).__init__()
                self.block1 = nn.Sequential(
                                nn.Conv2d(input_channel, output_channel, (kernel_size,1), stride, padding=(padding,0), dilation=dilation),
                                nn.Conv2d(output_channel, output_channel, (1,kernel_size), stride, padding=(0,padding), dilation=dilation),
                )
                self.block2 = nn.Sequential(
                                nn.Conv2d(input_channel, output_channel, (1,kernel_size), stride, padding=(0,padding), dilation=dilation),
                                nn.Conv2d(output_channel, output_channel, (kernel_size,1), stride, padding=(padding,0), dilation=dilation),
                )
                self.relu = nn.ReLU(inplace=True)
        def forward(self,x):
                branch1 = self.block1(x)
                branch2 = self.block2(x)
                result = self.relu(branch1+branch2)
                return result

class resblock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, atrous=1, downsample=None):
        super(resblock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, outplanes//self.expansion, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(outplanes//self.expansion, momentum=0.0003)
        self.conv2 = nn.Conv2d(outplanes//self.expansion, outplanes//self.expansion, kernel_size=3, stride=stride, padding=1*atrous, dilation=atrous, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(outplanes//self.expansion, momentum=0.0003)
        self.conv3 = nn.Conv2d(outplanes//self.expansion, outplanes, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(outplanes, momentum=0.0003)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.cut = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        elif self.inplanes != self.outplanes:
            residual = self.cut(x)

        out += residual
        out = self.relu(out)

        return out
