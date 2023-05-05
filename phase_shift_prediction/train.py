import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset

import torch
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm

parser = argparse.ArgumentParser(description='')

parser.add_argument('--init_model_file', default='', help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--data_dir', default='../data', help='Data folder', dest='data_dir')
parser.add_argument('--train_data_file', default='data_train_normalized.txt', help='train data', dest='train_data_file')
parser.add_argument('--val_data_file', default='data_valid_normalized.txt', help='validation data', dest='val_data_file')
parser.add_argument('--model_arch', default='4x8x4x1', help='NN model architecture: 4x8x8x4x1, 4x8x4x1, or 4x8x1', dest='model_arch')
parser.add_argument('--batch_size', default='1024', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='3e-4', type=float, help='Learning rate', dest='learning_rate')
parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', default=1000, type=int, help='Model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_file', default='loss_data', help='Text file to write step, loss metrics', dest='metrics_file')
parser.add_argument('--model_dir', default='saved_models', help='Directory to save models', dest='model_dir')


FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

if not os.path.exists(FLAGS.metrics_file):
	os.makedirs(FLAGS.metrics_file)

if not os.path.exists(FLAGS.model_dir):
	os.makedirs(FLAGS.model_dir)

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = '{}/step_loss_metrics{}.txt'.format(FLAGS.metrics_file, current_time)

for key in FLAGS_dict.keys():
	print('{} = {}'.format(key, FLAGS_dict[key]))


print('Preparing training dataset ...')
train_dataset = Dataset(data_dir=FLAGS.data_dir, data_file=FLAGS.train_data_file)
num_samples_train = train_dataset.num_samples
print("Training Data - num_samples: {}".format(num_samples_train))

print('Preparing validation dataset ...')
val_dataset = Dataset(data_dir=FLAGS.data_dir, data_file=FLAGS.val_data_file)
num_samples_val = val_dataset.num_samples
print("Validation Data - num_samples: {}".format(num_samples_val))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)

# get the model
model = Model(model_arch = FLAGS.model_arch)

# push the model to the right device
# if you have a GPU, the model will be pushed into GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# define criterion/loss function
criterion = torch.nn.MSELoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)

# initialize the model weights from a file
# can be used to continue previous training
# or for transfer learning
if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file)
		model.load_state_dict(state_dict['model_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))


# book-keeping
with open(metrics_file,'w') as f_metric_file:
	f_metric_file.write('# Model parameters:\n')

	for key in FLAGS_dict.keys():
		f_metric_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

	f_metric_file.write("# Training Data - num_samples: {}\n".format(num_samples_train))
	f_metric_file.write("# Validation Data - num_samples: {}\n".format(num_samples_val))
	
	f_metric_file.write('# epoch\ttraining_loss\tvalidation_loss\n')


min_validation_loss = 1000000
best_models_file = FLAGS.model_dir + "/best_models" + current_time + ".txt"
with open(best_models_file,'w') as f_best_models_file:
		f_best_models_file.write("# epoch\ttraining_loss\tvalidation_loss\n")

for epoch in range(FLAGS.num_epochs):
	# print('############## EPOCH - {} ##############'.format(epoch+1))
	training_loss = 0
	validation_loss = 0

	# train for one epoch
	# print('******** training ********')
	
	num_predictions = 0

	pbar = tqdm(total=len(train_data_loader))
	
	# set the model into training mode
	model.train()

	# iterate over mini-batches
	for samples, targets in train_data_loader:

		# push the mini-batch of data into device
        # if you have a GPU, the data will be pushed into GPU
		samples = samples.to(device)
		targets = targets.to(device)

		# 1. forward pass
		y_logits = model(samples)

		# 2. calculate the loss
		loss = criterion(y_logits, targets)

		# 3. clear the gradients
		optimizer.zero_grad()

		# 4. backward pass (back-propagation)
		# to compute gradients of parameters
		loss.backward()

		# 5. update the parameters
		optimizer.step()

		# book-keeping
		training_loss += loss.item()*targets.size(0)

		num_predictions += targets.size(0)

		pbar.update(1)


	training_loss /= num_predictions

	pbar.close()


	# evaluate on the validation dataset
	# print('******** validation ********')

	num_predictions = 0

	pbar = tqdm(total=len(val_data_loader))

	# set the model into evaluation mode
	# do not keep track of gradients 
	model.eval()
	with torch.no_grad():
		for samples, targets in val_data_loader:

			# push the mini-batch of data into device
			# if you have a GPU, the data will be pushed into GPU
			samples = samples.to(device)
			targets = targets.to(device)

			# inference
			y_logits = model(samples)

			# calculate the loss
			loss = criterion(y_logits, targets)

			# book-keeping
			validation_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)

			pbar.update(1)

	validation_loss /= num_predictions

	pbar.close()

	print('Epoch=%d ### training_loss=%.3e ### validation_loss=%.3e' % (epoch+1, training_loss, validation_loss))

	with open(metrics_file,'a') as f_metric_file:
		f_metric_file.write('%d\t%.3e\t%.3e\n' % (epoch+1, training_loss, validation_loss))


	# save model weights
	if (epoch+1) % FLAGS.save_interval == 0:
		model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
		state_dict = {'model_state_dict': model.state_dict()}
		torch.save(state_dict, model_weights_filename)
		print("Model weights saved in file: ", model_weights_filename)


	# save the best model weights
	if validation_loss < min_validation_loss:
			if validation_loss < min_validation_loss:
				min_validation_loss = validation_loss

			model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__best_' + str(epoch+1) + ".pth"
			state_dict = {  'model_state_dict': model.state_dict()}
			torch.save(state_dict, model_weights_filename)
			print("Best model weights saved in file: ", model_weights_filename)

			with open(best_models_file,'a') as f_best_models_file:
					f_best_models_file.write('%d\t%.3e\t%.3e\n' % (epoch+1, training_loss, validation_loss))


print('Training finished!!!')

# save model weights at the end of training
model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
state_dict = {	'model_state_dict': model.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)

