import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data_dir=None, data_file=None):

		self._data_dir = data_dir
		self._data_file = data_file

		# load data
		self._samples, self._labels = self.prepare_data()

		self._num_samples = len(self._samples)



	@property
	def num_samples(self):
		return self._num_samples

	def __len__(self):
		return self._num_samples


	def prepare_data(self):
		data_file_path = '{}/{}'.format(self._data_dir,self._data_file)

		# data file header
		# Vin1, Iin1, Vin2, Iin2, Theta
		data = np.loadtxt(data_file_path, delimiter='\t', comments='#', dtype=float)

		samples = torch.as_tensor(data[:,:4], dtype=torch.float32)
		labels = torch.as_tensor(data[:,4:5], dtype=torch.float32)
		# print('labels.shape: {}'.format(labels.shape))

		return samples, labels


	def __getitem__(self, idx):

		temp_sample = self._samples[idx]
		temp_label = self._labels[idx]

		return temp_sample, temp_label

