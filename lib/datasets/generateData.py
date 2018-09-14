# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.VOCDataset import VOCDataset
from datasets.COCODataset import COCODataset

def generate_dataset(dataset_name, cfg, period):
	if dataset_name == 'voc2012' or dataset_name == 'VOC2012':
		return VOCDataset('VOC2012', cfg, period)
	elif dataset_name == 'coco2017' or dataset_name == 'COCO2017':
		return COCODataset('COCO2017', cfg, period)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
