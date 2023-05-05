import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm


parser = argparse.ArgumentParser(description='')

parser.add_argument('--init_model_file', default='', help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--data_dir', default='../data', help='Data folder', dest='data_dir')
parser.add_argument('--data_file', default='data_test_normalized.txt', help='test data', dest='data_file')
parser.add_argument('--model_arch', default='4x8x4x1', help='NN model architecture: 4x8x8x4x1, 4x8x4x1, or 4x8x1', dest='model_arch')
parser.add_argument('--batch_size', default='1024', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--test_metrics_dir', default='test_metrics', help='Text file to write test metrics', dest='test_metrics_dir')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

print('Parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

print('Preparing dataset ...')
dataset = Dataset(data_dir=FLAGS.data_dir, data_file=FLAGS.data_file)
num_samples = dataset.num_samples
print("Data - num_samples: {}".format(num_samples))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)

# get the model
model = Model(model_arch = FLAGS.model_arch)

# push the model to the right device
# if you have a GPU, the model will be pushed into GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# initialize the model weights from a file
if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file, map_location=device)
		model.load_state_dict(state_dict['model_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))
	else:
		print('ERROR: model file ({}) does not exist!'.format(FLAGS.init_model_file))
		sys.exit()
else:
	print('ERROR: no model files provided!')
	sys.exit()


# create test metrics folder
model_name = FLAGS.init_model_file.split('__')[1] + '__' + FLAGS.init_model_file.split('__')[2] + '__' + FLAGS.init_model_file.split('__')[3][:-4]
data_folder_path = '{}/{}/{}'.format(FLAGS.test_metrics_dir,model_name,FLAGS.data_file[:-4])
if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)


# create test metrics file
test_metrics_filename = '{}/predictions_{}.txt'.format(data_folder_path,model_name)
with open(test_metrics_filename,'w') as f_metric_file:
	f_metric_file.write('# Parameters:\n')

	for key in FLAGS_dict.keys():
		f_metric_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

	f_metric_file.write('# Vin1\tIin1\tVin2\tIin2\tTheta\tTheta_pred\n')


# set the model into evaluation mode
# do not keep track of gradients 
model.eval()
with torch.no_grad():

	pbar = tqdm(total=len(data_loader))
	for samples, targets in data_loader:
		
		# push the mini-batch of data into device
		# if you have a GPU, the data will be pushed into GPU
		samples = samples.to(device)
		targets = targets.to(device)

		# get logits from model
		batch_logits = model(samples)

		num_points = targets.size(0)
		# print('num_points: {}'.format(num_points))


		batch_samples_arr = np.asarray(samples.cpu().numpy(),dtype=float)
		batch_truths_arr = np.asarray(targets.cpu().numpy(),dtype=float)
		batch_preds_arr = np.asarray(batch_logits.cpu().numpy(),dtype=float)

		pbar.update(1)


		for n in range(num_points):
			temp_Vin1 = batch_samples_arr[n][0]
			temp_Iin1 = batch_samples_arr[n][1]
			temp_Vin2 = batch_samples_arr[n][2]
			temp_Iin2 = batch_samples_arr[n][3]
			temp_Theta = batch_truths_arr[n][0]
			temp_Theta_pred = batch_preds_arr[n][0]

			with open(test_metrics_filename,'a') as f_metric_file:
				f_metric_file.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(temp_Vin1,temp_Iin1,temp_Vin2,temp_Iin2,temp_Theta,temp_Theta_pred))
				
			
	pbar.close()





