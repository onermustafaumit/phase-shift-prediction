import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

class Model(nn.Module):

	def __init__(self, model_arch):
		super(Model, self).__init__()


		if model_arch == "4x8x8x4x1":
			self._mlp = nn.Sequential(
										nn.Linear(4, 8),
										nn.ReLU(),
										nn.Linear(8, 8),
										nn.ReLU(),
										nn.Linear(8, 4),
										nn.ReLU(),
										nn.Linear(4, 1)
									)
		
		elif model_arch == "4x8x4x1":
			self._mlp = nn.Sequential(
										nn.Linear(4, 8),
										nn.ReLU(),
										nn.Linear(8, 4),
										nn.ReLU(),
										nn.Linear(4, 1)
									)
		
		elif model_arch == "4x8x1":
			self._mlp = nn.Sequential(
										nn.Linear(4, 8),
										nn.ReLU(),
										nn.Linear(8, 1)
									)

		else:
			print('ERROR: undefined architecture!')
			sys.exit()


		# initialize weights
		for m in self.modules():
			if isinstance(m, (nn.Linear)):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		out = self._mlp(x)

		return out

