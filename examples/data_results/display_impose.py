import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>



link_type = 1 #2
reward_type =  'dense' #'sparse'
length = 170 #100 #
max_torque =  45 #15 25 45 55
# sampling_type = 'Units64' #'O' #'Uniform' #'Units64' 'Varian2' 'Xavier' 'Bias'
sampling_type =   'Units256' #'Bias' #'0' #'Uniform' #'Units64' #'Varian2' #'Xavier' #'Bias' #'Xavier_Uniform' #'Units256_Xavier'

if length == 170: fix_length = 5
else: fix_length = 0 

if link_type == 1 and length == 100:
    path1 = '{}_link_{}_{}_dataset_total_T15.npy'.format(link_type,reward_type,length)
    path2 = '{}_link_{}_{}_dataset_total_T25.npy'.format(link_type,reward_type,length)
    path3 = '{}_link_{}_{}_dataset_total_T35.npy'.format(link_type,reward_type,length)
    path4 = '{}_link_{}_{}_dataset_total_T45.npy'.format(link_type,reward_type,length)
    path5 = '{}_link_{}_{}_dataset_total_T55.npy'.format(link_type,reward_type,length)
    exercise_label = 'varying_torque'
elif link_type == 1 and length == 170:
    path1 = '{}_link_{}_{}_dataset_total.npy'.format(link_type,reward_type,length) #['Default','w/ Bias','Variance','Units64','Uniform']
    path2 = '{}_link_{}_{}_dataset_total_Bias.npy'.format(link_type,reward_type,length)
    path3 = '{}_link_{}_{}_dataset_total_Varian2.npy'.format(link_type,reward_type,length)
    path4 = '{}_link_{}_{}_dataset_total_Units64.npy'.format(link_type,reward_type,length)
    path5 = '{}_link_{}_{}_dataset_total_Units256.npy'.format(link_type,reward_type,length)
    path6 = '{}_link_{}_{}_dataset_total_Uniform.npy'.format(link_type,reward_type,length)
    
    path7 = '{}_link_{}_{}_dataset_total_Xavier.npy'.format(link_type,reward_type,length)
    path8 = '{}_link_{}_{}_dataset_total_Xavier_Uniform.npy'.format(link_type,reward_type,length)
    path9 = '{}_link_{}_{}_dataset_total_Units256_Xavier.npy'.format(link_type,reward_type,length)
    exercise_label = 'varying_initialisation'
else: print('Incorrect Tasks!')

# print('path: ', path)
# xxx
# data = np.load(path)

if exercise_label == 'varying_torque':
    list_data = [np.load(path1),np.load(path2),np.load(path3),np.load(path4),np.load(path5)]
else:
    ################# w/0 Xavier #############################
    list_data = [np.load(path1),np.load(path2),np.load(path3),
                 np.load(path4),np.load(path5),np.load(path6)] 
    
    ################# w/ Xavier #############################
    # list_data = [np.load(path1),np.load(path2),np.load(path3),
    #              np.load(path4),np.load(path5),np.load(path6),
    #              np.load(path7),np.load(path8),np.load(path9)] 

list_ordered_means_df = []
list_ordered_std_df = []
list_ordered_rank_df = []
for data in list_data:
    # print(min([i.shape[0] for i in list_data]))
    # xxxx
    cap = min([i.shape[0] for i in list_data])
    samples_ = pd.DataFrame(data)[:cap] #returns_per_sample
    # print('samples_: ', samples_.shape)
    # print('samples_: ', samples_[:800].shape)
    ####################### Normalization of Returns ##############################
    # global_min = samples_.min().min()
    # global_max = samples_.max().max()

    # samples_scaled = (samples_ - global_min)/(global_max - global_min) 
    # samples_ = samples_scaled #scaled_values
    ####################### Normalization of Returns ##############################

    samples_df = samples_ #[:10] #returns_per_sample
    stats_df = samples_df.transpose().describe() #statistical summary 
    means_df = stats_df.loc['mean'] #means_of_samples
    std_df = stats_df.loc['std']

    check = np.mean(samples_df, axis=1) #correct_means
    order_check = sorted(check)

    v_samples = samples_df.transpose() #swapped rows_n_cols
    sorted_cols = sorted(v_samples.columns, key=lambda col: v_samples[col].mean())
    ordered_samples = v_samples[sorted_cols]
    mastered = ordered_samples.melt(value_name='scores')

    # x = mastered['variable']
    # y = mastered['scores']
    # mask = np.array(y) > 2 #0.98 #np.array(y) > -10.0

    summary_data_df = pd.DataFrame()
    summary_data_df['means'] = means_df
    summary_data_df['stds'] = std_df
    summary_data_df['rank'] = summary_data_df['means'].rank()
    order_summary_data_df = summary_data_df[['means','stds','rank']].sort_values(by='means').reset_index(drop=True) #.to_dict('index') #    

    ordered_means_df = order_summary_data_df['means'].to_numpy() #means_of_samples
    ordered_std_df = order_summary_data_df['stds'].to_numpy()
    ordered_var_df = ordered_std_df**2
    ordered_rank_df = order_summary_data_df['rank'].to_numpy()

    list_ordered_means_df.append(ordered_means_df)
    list_ordered_std_df.append(ordered_std_df)
    list_ordered_rank_df.append(ordered_rank_df)

# xxxx

# print(len(list_ordered_means_df))
# xxx

## plotting
val_size = 6
f_size = 16
figsize = [val_size,val_size]

dense_lim = -500

if exercise_label == 'varying_torque':
    colors = ["red", "green" ,"blue" ,"orange", 'gray']
    names = ['T15', 'T25', 'T35', 'T45', 'T55'] 
else:
    ############################################### w/o Xavier ############################################### 
    colors = ["red", "green" ,"blue" ,"orange", "mediumorchid", "grey"]
    names = ['Default','w/ Bias','Variance','Units64','Units256','Uniform']

    ############################################### w/ Xavier ############################################### 
    # names = ['Default','w/ Bias','Variance','Units64', 'Units256','Uniform','Xavier','Xavier_Uniform','Units256_Xavier']
    # colors = ["red", "green" ,"blue" ,"orange", "mediumorchid", "grey", "teal", "darkblue", "purple"]

plt.figure(figsize=figsize)

for i in range(len(list_ordered_means_df)):
    plt.scatter(list_ordered_means_df[i],list_ordered_std_df[i],color=colors[i],alpha=0.2,label=names[i]) #std_vs_mean

plt.ylabel('$\sqrt{V_{n}}$', fontsize=f_size)
plt.xlabel('$M_{n}$', fontsize=f_size)
plt.xticks(fontsize=f_size)
plt.yticks(fontsize=f_size)
plt.legend(fontsize=14)
plt.tight_layout()

if reward_type == 'sparse': plt.xlim(-55,0)
else: plt.xlim(-500,0)

# y_max = ordered_std_df.max() #+10
# print('y_max: ', y_max)
# xxx
# plt.ylim(-500,0)
if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
plt.savefig('std_dist_{}.png'.format(exercise_label), dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.figure(figsize=figsize)
for i in range(len(list_ordered_means_df)):
    plt.hist(list_ordered_means_df[i], log=True, edgecolor="black",color=colors[i],alpha=0.5,label=names[i]) #mean_histogram

plt.xlabel('$M_{n}$', fontsize=f_size)
plt.ylabel('log(Counts)', fontsize=f_size)
plt.xticks(fontsize=f_size)
plt.yticks(fontsize=f_size)
plt.legend(fontsize=14)
plt.tight_layout()

if reward_type == 'sparse': plt.xlim(-55,0)
else: plt.xlim(-500,0)

if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
plt.savefig('mean_hist_{}.png'.format(exercise_label), dpi=300, bbox_inches='tight')
# plt.show()
# xxx

plt.figure(figsize=figsize)
for i in range(len(list_ordered_means_df)):
    # plt.plot(x[mask],y[mask],'o',color='limegreen', alpha=0.01, markersize=3)
    # plt.plot(x[~mask],y[~mask],'o',color='tomato', alpha=0.01, markersize=3)
    plt.plot(list_ordered_rank_df[i],list_ordered_means_df[i], color=colors[i],alpha=1,label=names[i]) #mean_vs_rank
plt.ylabel('$S_{a,n,e}$ and $M_{n}$',fontsize=f_size)
plt.xlabel('$R_{n}$', fontsize=f_size)
plt.xticks(fontsize=f_size)
plt.yticks(fontsize=f_size)
plt.legend(fontsize=14)
plt.tight_layout()

if reward_type == 'sparse': plt.ylim(-55,0)
else: plt.ylim(-500,0)

if link_type == 1: plt.title("{}-link arm ({})_{}".format(link_type,reward_type,length-fix_length), fontsize=f_size)
else:    plt.title("{}-link arm ({})".format(link_type,reward_type), fontsize=f_size)
plt.savefig('mean_curve.png_{}.png'.format(exercise_label), dpi=300, bbox_inches='tight')
plt.show()
xxx

plt.tight_layout()
# if link_type == 1: plt.savefig('{}_link_{}_n10200_{}.png'.format(link_type,reward_type,length), dpi=300, bbox_inches='tight')
# else:    plt.savefig('{}_link_{}_n10200.png'.format(link_type,reward_type), dpi=300, bbox_inches='tight')

if link_type == 1: plt.savefig('{}_link_{}_norm_{}.png'.format(link_type,reward_type,length), dpi=300, bbox_inches='tight')
else:    plt.savefig('{}_link_{}_norm.png'.format(link_type,reward_type), dpi=300, bbox_inches='tight')

plt.show()