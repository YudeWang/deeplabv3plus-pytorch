import torch.nn as nn
import math
import torchvision.models as models

class ResNet(nn.modules):
	
	def __init__(self, layers, atrous, pretrained=True):
		super(ResNet, self).__init__()
		self.inner_layer = []
		if layers == 18:
			self.backbone = models.resnet18(pretrained=pretrained)
		elif layers == 34:
			self.backbone = models.resnet34(pretrained=pretrained)
		elif layers == 50:
			self.backbone = models.resnet50(pretrained=pretrained)
		elif layers == 101:
			self.backbone = models.resnet101(pretrained=pretrained)
		elif layers == 152:
			self.backbone = models.resnet152(pretrained=pretrained)
		else:
			raise ValueError('resnet.py: network layers is no support yet')
		
		def hook_func(module, input, output):
			self.inner_layer.append(output)

		self.backbone.layer1.register_forward_hook(hook_func)	
		self.backbone.layer2.register_forward_hook(hook_func)
		self.backbone.layer3.register_forward_hook(hook_func)
		self.backbone.layer4.register_forward_hook(hook_func)

	def forward(self,x):
		self.inner_layer = []
		
