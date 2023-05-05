import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})

def moving_avg_filter(data_arr, w):
	data_arr_cumsum = np.cumsum(data_arr)
	data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
	data_arr_filtered = data_arr_cumsum[w-1:]/w

	return data_arr_filtered

parser = argparse.ArgumentParser(description='Plot the loss vs iteration and accuracy vs iteration for given data file')
parser.add_argument('--data_file', help='Data file path', dest='data_file')
parser.add_argument('--step_size', default=1, type=int, help='Data file path', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Data file path', dest='filter_size')
FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')

steps = np.arange(data_arr.shape[0]) + 1
train_loss = data_arr[:,1]
val_loss = data_arr[:,2]

if w>1:
	steps = steps[w-1:]
	train_loss = moving_avg_filter(train_loss,w)
	val_loss = moving_avg_filter(val_loss,w)

ind_start = 0
ind_step = FLAGS.step_size
ind_end = len(steps)

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], 'r', label="train")
ax.plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], 'b', label="valid")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='--')
ax.legend()


fig.tight_layout()

fig_filename = '{}.pdf'.format(FLAGS.data_file[:-4])
fig.savefig(fig_filename, dpi=300)
fig_filename = '{}.png'.format(FLAGS.data_file[:-4])
fig.savefig(fig_filename, dpi=300)

plt.show()
	
