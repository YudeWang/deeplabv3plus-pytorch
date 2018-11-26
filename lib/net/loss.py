#=========================================
# Written by Yude Wang
#=========================================
import torch
import torch.nn as nn
class MaskLoss(nn.Module):
	def __init__(self,reduction):
		super(MaskLoss, self).__init__()
		self.loss = None
		self.reduction = reduction
	def forward(self, x, y, mask):
		if self.loss == None:
			raise ValueError('loss.py: MaskLoss.loss has not been implemented')
		count = torch.sum(mask)
		loss = self.loss(x,y)
		loss = loss * mask
		if self.reduction == 'all':
			return torch.sum(loss)/count
		elif self.reduction == 'none':
			return loss

class MaskCrossEntropyLoss(MaskLoss):
	def __init__(self,reduction='all'):
		super(MaskCrossEntropyLoss, self).__init__(reduction)
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

class MaskBCELoss(MaskLoss):
	def __init__(self,reduction='all'):
		super(MaskBCELoss, self).__init__(reduction)
		self.loss = torch.nn.BCELoss(reduction='none')

class MaskBCEWithLogitsLoss(MaskLoss):
	def __init__(self,reduction='all'):
		super(MaskBCEwithLogitsLoss, self).__init__(reduction)
		self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
