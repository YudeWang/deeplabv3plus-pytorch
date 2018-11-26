# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.VOCDataset import VOCDataset
from datasets.COCODataset import COCODataset
from datasets.ADE20KDataset import ADE20KDataset
from datasets.ContextDataset import ContextDataset
from datasets.CityscapesDataset import CityscapesDataset

def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'voc2012' or dataset_name == 'VOC2012':
		return VOCDataset('VOC2012', cfg, period, aug)
	elif dataset_name == 'coco2017' or dataset_name == 'COCO2017':
		return COCODataset('COCO2017', cfg, period)
	elif dataset_name == 'ade20k' or dataset_name == 'ADE20K':
		return ADE20KDataset('ADE20K', cfg, period)
	elif dataset_name == 'context' or dataset_name == 'Context':
		return ContextDataset('Context', cfg, period)
	elif dataset_name == 'cityscapes' or dataset_name == 'Cityscapes':
		return CityscapesDataset('Cityscapes', cfg, period)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
