import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>
data = np.load('two_link_total.npy')

print(data.shape)

        
samples_df = pd.DataFrame(data) #returns_per_sample
stats_df = samples_df.transpose().describe() #statistical summary 
means_df = stats_df.loc['mean'] #means_of_samples
std_df = stats_df.loc['std']

summary_data_df = pd.DataFrame()
summary_data_df['means'] = means_df

z = means_df #[:10]
# print(z)
# print(np.array(means_df))
# print(z.sort_values(inplace=True))


# xxx
# print(samples_df)
# flatten = np.concatenate(np.array(samples_df))
# print(flatten)
# print(flatten.shape)


z1 = z.sort_values()
# print(z1)
# print(z1.shape[0])

index = np.arange(1,z1.shape[0]+1)
# print(index)
# print(np.flip(index))
# xxx

z2 = pd.DataFrame()
z2['mean'] = z1
z2['rank'] = index
z3 = np.array(samples_df[0].sort_values())
# print(z2)
# xxx
# print(z3.shape)
# xxx
# i = 0
# for i in range(1):
#     z3 = np.array(samples_df[i].sort_values())
#     # print(z3)
#     z2['ex_{}'.format(i)] = z3
# print(z2)
# xxx

# xx
# print(means_df.rank())
# xxx

# # xxx

summary_data_df['stds'] = std_df
summary_data_df['rank'] = summary_data_df['means'].rank()


# print(std_df)
# print(np.array(means_df).flatten())
# xxx


# print(summary_data_df['rank'])

order_summary_data_df = summary_data_df.sort_values(by='rank')
ordered_means = order_summary_data_df['means'].values

# print(order_summary_data_df)
# print(order_summary_data_df['means'].values)
# xxx

# print(order_summary_data_df)
# xxx

# trans_samples_df = samples_df.T
# trans_samples_df['means'] = means_df
# trans_samples_df['stds'] = std_df
# trans_samples_df['rank'] = trans_samples_df['means'].rank()
# order_trans_samples_df = trans_samples_df.sort_values(by='rank')

# rearrange_df = order_trans_samples_df.rename(dict(zip(order_trans_samples_df.index,order_trans_samples_df['rank']))) #rename rows according to column values
# sorted_rank_data_df = rearrange_df.drop(['means', 'stds', 'rank'], axis=1) #drop means,stds,rank | 

#print(samples_df)
#print(stats_df)
#print(means_df)
#print(std_df)
#print(summary_data_df)
#print(order_summary_data_df)
#print(trans_samples_df)
#print(order_trans_samples_df)
#print(rearrange_df)
#print(sorted_rank_data_df)

#save to csv
# samples_df.to_csv('data_1link_RWG_a',encoding='utf-8',index=False)


##plot std vs means
#summary_data_df.plot(kind='scatter', x='means', y='std') #std_vs_mean
"""
plt.subplot(2,1,1)
plt.scatter(summary_data_df['means'],summary_data_df['stds']) #std_vs_mean

plt.subplot(2,2,3)
plt.hist(summary_data_df['means'], log=True) #mean_histogram

plt.subplot(2,2,2)
plt.plot(sorted_rank_data_df, 'rx')

plt.subplot(2,2,4)
#plt.scatter(summary_data_df['rank'],summary_data_df['means']) #mean_vs_rank
plt.plot(sorted_rank_data_df, 'rx')
plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'b') #mean_vs_rank
"""

#order_trans_samples_df.plot(x='rank',y=['means','stds'])
#plt.scatter(order_summary_data_df['rank'],order_summary_data_df['means'])
#plt.plot(rearr2, 'o')
val_size = 3
figsize = [3*val_size,val_size]
plt.figure(figsize=figsize)

means_ing = np.array(means_df).flatten()
stds_ing = np.array(std_df).flatten()

plt.subplot(1,3,3)
# plt.scatter(summary_data_df['means'],summary_data_df['stds'],"o", color="mediumorchid",alpha=0.2) #std_vs_mean
plt.scatter(means_ing,stds_ing ,color="mediumorchid",alpha=0.1) #std_vs_mean
plt.ylabel('std')
plt.xlabel('mean')
plt.xlim(-500,0)
plt.title("2-link arm (dense)")
# plt.savefig('std_vs_mean.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

# plt.figure(figsize=figsize)
plt.subplot(1,3,1)
plt.hist(summary_data_df['means'], log=True, edgecolor="black") #mean_histogram
plt.xlabel('mean')
plt.ylabel('log(Counts)')
plt.xlim(-500,0)
plt.title("2-link arm (dense)")
# plt.savefig('mean_hist.png', dpi=300, bbox_inches='tight')
# plt.show()
# xxx

# plt.figure(figsize=figsize)
# plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'b') #mean_vs_rank
# plt.plot(z2['rank'],z2['mean'], 'rx') #returns_vs_rank

z3 = np.array(samples_df[120].sort_values())
r3 = np.arange(1,z3.shape[0]+1) 

print(r3)
print(order_summary_data_df['rank'])


testy = pd.DataFrame()
testy['mean'] = z3
testy['rank'] = r3
testy['ordered'] = ordered_means
print(testy[:-5])

# xx

plt.subplot(1,3,2)

# for i in range(23,50):
#     z3 = np.array(samples_df[i].sort_values())
#     r3 = np.arange(1,z3.shape[0]+1) 
#     testy = pd.DataFrame()
#     testy['mean'] = z3
#     testy['rank'] = r3
#     testy['ordered'] = ordered_means
#     plt.plot(testy['rank'],testy['mean'], 'ro',alpha=0.1) #mean_vs_rank


plt.plot(testy['rank'],testy['ordered'], 'k') #mean_vs_rank
# plt.plot(order_summary_data_df['rank'],order_summary_data_df['means'], 'k') #mean_vs_rank
plt.ylabel('score & mean')
plt.xlabel('rank')
plt.ylim(-500,0)
plt.title("2-link arm (dense)")
# plt.savefig('plots.png', dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.savefig('plots.png', dpi=300, bbox_inches='tight')
plt.show()