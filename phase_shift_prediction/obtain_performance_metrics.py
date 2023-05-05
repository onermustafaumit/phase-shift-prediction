import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import argparse

plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})

def rmse(data_arr1, data_arr2):
	error = data_arr1 - data_arr2
	rmse_val = np.sqrt(np.mean(error**2))

	return rmse_val

def mae(data_arr1, data_arr2):
	error = data_arr1 - data_arr2
	mae_val = np.mean(np.abs(error))

	return mae_val

def BootStrap(data_arr1, data_arr2, n_bootstraps, score_fnc):

	# control reproducibility
	rng_seed = 42  
	rng = np.random.RandomState(rng_seed)
	
	bootstrapped_scores = []
	for i in range(n_bootstraps):

		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(data_arr1), len(data_arr1))

		score = score_fnc(data_arr1[indices], data_arr2[indices])
		bootstrapped_scores.append(score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.

	# Computing the lower and upper bound of the 95% confidence interval
	# The bounds percentiles: 0.025 and 0.975
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

	return sorted_scores, confidence_lower, confidence_upper



parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_file', help='Data file path', dest='data_file')
FLAGS = parser.parse_args()


data_file = FLAGS.data_file
data_folder_path = '/'.join(data_file.split('/')[:-1])
model_name = data_file.split('/')[-1][12:-4]

# minimums and maximums for
# voltages, currents, and 
# phase shift angles
V_max = 50
V_min = 15
I_max = 3
I_min = 0.6
theta_max = 360
theta_min = 0

Vin1_max = V_max
Vin1_min = V_min
Iin1_max = I_max
Iin1_min = I_min
Vin2_max = V_max
Vin2_min = V_min
Iin2_max = I_max
Iin2_min = I_min

# read the normalized data and predictions
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=float)

Vin1_normalized = np.asarray(data[:,0], dtype=float)
Iin1_normalized = np.asarray(data[:,1], dtype=float)
Vin2_normalized = np.asarray(data[:,2], dtype=float)
Iin2_normalized = np.asarray(data[:,3], dtype=float)
theta_normalized = np.asarray(data[:,4], dtype=float)
theta_pred_normalized = np.asarray(data[:,5], dtype=float)

# convert them back to raw values
Vin1 = Vin1_normalized*(Vin1_max - Vin1_min) + Vin1_min
Iin1 = Iin1_normalized*(Iin1_max - Iin1_min) + Iin1_min
Vin2 = Vin2_normalized*(Vin2_max - Vin2_min) + Vin2_min
Iin2 = Iin2_normalized*(Iin2_max - Iin2_min) + Iin2_min
theta = theta_normalized*(theta_max - theta_min) + theta_min
theta_pred = theta_pred_normalized*(theta_max - theta_min) + theta_min

# calculate error
error = theta - theta_pred


##### get statistics #####
root_mean_square_error = np.sqrt(np.mean(error**2))
sorted_rmses, rmse_lower, rmse_upper = BootStrap(theta, theta_pred, n_bootstraps=10000, score_fnc=rmse)

mean_absolute_error = np.mean(np.abs(error))
sorted_maes, mae_lower, mae_upper = BootStrap(theta, theta_pred, n_bootstraps=10000, score_fnc=mae)

print('root_mean_square_error = {:.4f} (95% CI:{:.4f} - {:.4f})'.format(root_mean_square_error,rmse_lower,rmse_upper))
print('mean_absolute_error = {:.4f} (95% CI:{:.4f} - {:.4f})'.format(mean_absolute_error,mae_lower,mae_upper))

statistics_file = '{}/statistics__{}.txt'.format(data_folder_path,model_name)
with open(statistics_file,'w') as f_statistics_file:
	f_statistics_file.write('root_mean_square_error = {:.4f} (95% CI:{:.4f} - {:.4f})\n'.format(root_mean_square_error, rmse_lower, rmse_upper))
	f_statistics_file.write('mean_absolute_error = {:.4f} (95% CI:{:.4f} - {:.4f})\n'.format(mean_absolute_error, mae_lower, mae_upper))


##### cumulative percentages #####
error_sorted = np.sort(error)
num_data_points = len(error_sorted)
indices = np.arange(1,num_data_points+1)
percentages = 100*(indices/num_data_points)


# calculate % of data points falling having residual error 
# in the range of -0.25 < residual error < 0.25
percent_min = percentages[error_sorted > -0.25][0]
percent_max = percentages[error_sorted < 0.25][-1]

# plot residual error vs cumulative % of data points curve
fig1, ax1 = plt.subplots(1,figsize=(4,1.5))
ax1.plot(percentages, error_sorted, color='k')
ax1.axvline(percent_min, color='red', linestyle='--')
ax1.axvline(percent_max, color='red', linestyle='--')
ax1.annotate('', xy=(percent_min,-0.75), xytext=(percent_max,-0.75), arrowprops=dict(arrowstyle='<->',color='red'))
ax1.text(50,-0.75,'~{:d}%'.format(round(percent_max - percent_min)), horizontalalignment='center', verticalalignment='center', color='red', bbox={'boxstyle':'square', 'facecolor':'white', 'edgecolor':'white'})
ax1.set_ylim((-1.,1.))
ax1.set_yticks(np.arange(-1.,1.001,0.5))
ax1.set_ylabel(r'Error $(\degree)$')
ax1.set_xlabel('Cumulative % of data points in the test set')
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, which='major', color='lightgrey', linestyle='--', linewidth=1.)
ax1.yaxis.grid(True, which='minor', color='lightgrey', linestyle='-.', linewidth=.5)

fig1.tight_layout()
fig1.subplots_adjust(left=0.15, bottom=0.30, right=0.98, top=0.95, wspace=0.2, hspace=0.2)
fig1_filename = '{}/residual_errors__{}.pdf'.format(data_folder_path,model_name)
fig1.savefig(fig1_filename, dpi=300)
fig1_filename = '{}/residual_errors__{}.png'.format(data_folder_path,model_name)
fig1.savefig(fig1_filename, dpi=300)



plt.show()

