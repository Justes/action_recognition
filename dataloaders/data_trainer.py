import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from .dataset import VideoDataset

class Collator(object):

	def __init__(self, objective):
		self.objective = objective
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		mask_list = []
		marker_list = []
		for record in minibatch:
			image_list.append(record[0])
			label_list.append(record[1])
			if self.objective == 'mim':
				mask_list.append(record[2])
				marker_list.append(record[3])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		if self.objective == 'mim':
			minibatch.append(torch.stack(label_list))
			minibatch.append(torch.stack(mask_list))
			minibatch.append(marker_list)
		else:
			label = np.stack(label_list)
			minibatch.append(torch.from_numpy(label))
		
		return minibatch

class DataModule(pl.LightningDataModule):
	def __init__(self, 
				 configs,
				 dataset,
				 frame,
				 root,
				 ):
		super().__init__()
		self.train_dataset = VideoDataset(dataset=dataset, split='train', clip_len=frame, root=root)
		self.val_dataset = VideoDataset(dataset=dataset, split='test', clip_len=frame, root=root)
		self.configs = configs


	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			collate_fn=Collator(self.configs.objective).collate,
			shuffle=True,
			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			collate_fn=Collator(self.configs.objective).collate,
			shuffle=False,
			drop_last=False,
		)
