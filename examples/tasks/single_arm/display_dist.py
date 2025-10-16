import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>



link_type = 1 #2
reward_type = 'dense' 
data = np.load('{}_link_{}_dataset__50_total.npy'.format(link_type,reward_type))
        
samples_df = pd.DataFrame(data) #returns_per_sample
stats_df = samples_df.transpose().describe() #statistical summary 
means_df = stats_df.loc['mean'] #means_of_samples
std_df = stats_df.loc['std']

summary_data_df = pd.DataFrame()
summary_data_df['means'] = means_df
summary_data_df['stds'] = std_df
# print('summary_data_df: \n', summary_data_df)

summary_data_df['rank'] = summary_data_df['means'].rank()
# print('summary_data_df: \n', summary_data_df)

order_summary_data_df = summary_data_df[['means','stds','rank']].sort_values(by='means').reset_index(drop=True) #.to_dict('index') #    
# print('order_summary_data_df: \n', order_summary_data_df)

ordered_means_df = order_summary_data_df['means'].to_numpy() #means_of_samples
ordered_std_df = order_summary_data_df['stds'].to_numpy()
ordered_var_df = ordered_std_df**2
ordered_rank_df = order_summary_data_df['rank'].to_numpy()

## plotting
val_size = 3
f_size = 11
f_size = 9
figsize = [3*val_size,val_size]
plt.figure(figsize=figsize)

plt.subplot(1,3,3)
plt.scatter(ordered_means_df,ordered_std_df ,color="mediumorchid",alpha=0.1) #std_vs_mean
plt.ylabel('$\sqrt{V_{n}}$', fontsize=f_size)
plt.xlabel('$M_{n}$', fontsize=f_size)
plt.xlim(-500,0)
y_max = ordered_std_df.max()+10
plt.ylim(0,y_max)
plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
# plt.savefig('std_vs_mean.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.subplot(1,3,1)
plt.hist(ordered_means_df, log=True, edgecolor="black") #mean_histogram
plt.xlabel('$M_{n}$', fontsize=f_size)
plt.ylabel('log(Counts)', fontsize=f_size)
plt.xlim(-500,0)
plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
# plt.savefig('mean_hist.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.subplot(1,3,2)
plt.plot(ordered_rank_df,ordered_means_df, 'k') #mean_vs_rank
plt.ylabel('$M_{n}$',fontsize=f_size)
plt.xlabel('$R_{n}$', fontsize=f_size)
plt.ylim(-500,0)
plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)


plt.tight_layout()
plt.savefig('{}_link_{}_n5099.png'.format(link_type,reward_type), dpi=300, bbox_inches='tight')
plt.show()