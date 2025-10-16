import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>



link_type = 1 #2
reward_type =  'dense' #'sparse'
length = 100 #100 #
max_torque =  45 #15 25 45 55
sampling_type = 'Units64' #'O' #'Uniform' #'Units64' 'Varian2' 'Xavier' 'Bias'

if length == 170: fix_length = 5
else: fix_length = 0 

if link_type == 1: 
    path = '{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length)
    if length == 100 and max_torque != 0:
        path = '{}_link_{}_{}_dataset_total_T{}.npy'.format(link_type,reward_type,length,max_torque)
    elif length == 170 and max_torque != 0 and sampling_type != 'O':
        path = '{}_link_{}_{}_dataset_total_{}.npy'.format(link_type,reward_type,length,sampling_type)
else:
    path = '{}_link_{}_dataset_total.npy'.format(link_type,reward_type)

# print('path: ', path)
# xxx
data = np.load(path)
samples_ = pd.DataFrame(data) #returns_per_sample
# print('samples: \n', samples_)

####################### Normalization of Returns ##############################
global_min = samples_.min().min()
global_max = samples_.max().max()
# print('global_min: \n ', global_min)
# print('global_max: \n ', global_max)

samples_scaled = (samples_ - global_min)/(global_max - global_min)
# print('df_scaled: \n ', df_scaled)

# crown = df_scaled.to_numpy()
# # print('crown: ', crown.shape)
   

samples_ = samples_scaled #scaled_values
# print('samples: \n', samples_)
# xxx
####################### Normalization of Returns ##############################

# xxx
samples_df = samples_ #[:10] #returns_per_sample
stats_df = samples_df.transpose().describe() #statistical summary 
means_df = stats_df.loc['mean'] #means_of_samples
std_df = stats_df.loc['std']

check = np.mean(samples_df, axis=1) #correct_means
order_check = sorted(check)

# print(stats_df)
# check = np.mean(samples_df, axis=1) #correct_means
# order_check = sorted(check)
# print('means_df: \n', means_df)
# print('check: \n', check)
# print('diff: \n', check - means_df)
# xxx

v_samples = samples_df.transpose() #swapped rows_n_cols
sorted_cols = sorted(v_samples.columns, key=lambda col: v_samples[col].mean())
ordered_samples = v_samples[sorted_cols]
mastered = ordered_samples.melt(value_name='scores')

# print('v_samples: \n', v_samples)
# print('sorted_cols: \n', sorted_cols)
# print('shape:',v_samples.shape[0])
# print('order_sorted_cols: \n', order_sorted_cols)
# print('ordered_samples: \n', ordered_samples.values.flatten())
# print('mastered: \n', mastered)

x = mastered['variable']
y = mastered['scores']
mask = np.array(y) > 2 #0.98 #np.array(y) > -10.0

# plt.plot(mastered['variable'],mastered['scores'],'o',color='tomato', alpha=1, markersize=3)
# plt.plot(x[mask],y[mask],'o',color='limegreen', alpha=0.01, markersize=3)
# plt.plot(x[~mask],y[~mask],'o',color='tomato', alpha=0.01, markersize=3)
# plt.show()
# xxx


summary_data_df = pd.DataFrame()
summary_data_df['means'] = means_df
summary_data_df['stds'] = std_df
# print('summary_data_df: \n', summary_data_df)

summary_data_df['rank'] = summary_data_df['means'].rank()
# print('summary_data_df: \n', summary_data_df)

order_summary_data_df = summary_data_df[['means','stds','rank']].sort_values(by='means').reset_index(drop=True) #.to_dict('index') #    
# print('order_summary_data_df: \n', order_summary_data_df)
# xxx

ordered_means_df = order_summary_data_df['means'].to_numpy() #means_of_samples
ordered_std_df = order_summary_data_df['stds'].to_numpy()
ordered_var_df = ordered_std_df**2
ordered_rank_df = order_summary_data_df['rank'].to_numpy()

## plotting
val_size = 3
f_size = 11
figsize = [3*val_size,val_size]

dense_lim = -500

plt.figure(figsize=figsize)

plt.subplot(1,3,3)
plt.scatter(ordered_means_df,ordered_std_df ,color="mediumorchid",alpha=0.1) #std_vs_mean
plt.ylabel('$\sqrt{V_{n}}$', fontsize=f_size)
plt.xlabel('$M_{n}$', fontsize=f_size)

if reward_type == 'sparse': plt.xlim(-0.1,1.1) #plt.xlim(-55,0)
else: plt.xlim(-0.1,1.1) #plt.xlim(dense_lim,0)

y_max = ordered_std_df.max() #+10
# print('y_max: ', y_max)
# xxx
plt.ylim(-0.1,1.1) # plt.ylim(-0.5,y_max)
if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
# plt.savefig('std_vs_mean.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.subplot(1,3,1)
plt.hist(ordered_means_df, log=True, edgecolor="black") #mean_histogram
plt.xlabel('$M_{n}$', fontsize=f_size)
plt.ylabel('log(Counts)', fontsize=f_size)

if reward_type == 'sparse': plt.xlim(-0.1,1.1) #plt.xlim(-55,0)
else: plt.xlim(-0.1,1.1) #plt.xlim(dense_lim,0)

if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
# plt.savefig('mean_hist.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.subplot(1,3,2)

plt.plot(x[mask],y[mask],'o',color='limegreen', alpha=0.01, markersize=3)
plt.plot(x[~mask],y[~mask],'o',color='tomato', alpha=0.01, markersize=3)
plt.plot(ordered_rank_df,ordered_means_df, 'k') #mean_vs_rank
plt.ylabel('$S_{a,n,e}$ and $M_{n}$',fontsize=f_size)
plt.xlabel('$R_{n}$', fontsize=f_size)

if reward_type == 'sparse': plt.ylim(-0.1,1.1) #plt.ylim(-55,0)
else: plt.ylim(-0.1,1.1) # plt.ylim(dense_lim,0)

if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)


plt.tight_layout()
# if link_type == 1: plt.savefig('{}_link_{}_n10200_{}.png'.format(link_type,reward_type,length), dpi=300, bbox_inches='tight')
# else:    plt.savefig('{}_link_{}_n10200.png'.format(link_type,reward_type), dpi=300, bbox_inches='tight')

if link_type == 1: plt.savefig('{}_link_{}_norm_{}.png'.format(link_type,reward_type,length), dpi=300, bbox_inches='tight')
else:    plt.savefig('{}_link_{}_norm.png'.format(link_type,reward_type), dpi=300, bbox_inches='tight')

plt.show()