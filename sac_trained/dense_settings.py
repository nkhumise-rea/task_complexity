import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

#modules
def roll_mean(array,window=2):
    return np.convolve(array,np.ones(window)/window, mode="same")

def plotting(grid,mean,std):
        ## Plots
        #fig = plt.figure(figsize=[17,15])
        plt.figure(figsize=[17,15])
        plt.plot(grid, mean,color="tomato",label='agent')
        plt.fill_between(grid,
                        mean - std,
                        mean + std,
                        facecolor="tomato", 
                        alpha=0.15,
                        )      
        plt.legend()
        plt.xlabel('Step')
        plt.grid()
        plt.ylabel('Return')
        # plt.title("results")
        # # save the figure
        # plt.savefig(
        #     'eval_trained_plots.png', 
        #     dpi=300, 
        #     bbox_inches='tight',
        #     format='png')
        plt.show()
        return 

def group_plot(grids,means,stds):
    plt.figure(figsize=[8,6]) # plt.figure(figsize=[17,15])
    f_size = 16

    plt.plot(grids[0],means[0],color="red",label='1link_100')
    plt.fill_between(grids[0],
                    means[0] - stds[0],
                    means[0] + stds[0],
                    facecolor="red", 
                    alpha=0.1,
                    )  
    
    plt.plot(grids[1],means[1],color="deepskyblue",label='1link_165')
    plt.fill_between(grids[1],
                    means[1] - stds[1],
                    means[1] + stds[1],
                    facecolor="deepskyblue", 
                    alpha=0.1,
                    )  
    
    plt.plot(grids[2],means[2],color="purple",label='2link')
    plt.fill_between(grids[2],
                    means[2] - stds[2],
                    means[2] + stds[2],
                    facecolor="purple", 
                    alpha=0.1,
                    )  

    
    plt.legend(loc='lower right',fontsize=f_size)
    
    # plt.yscale('symlog')
    # plt.xscale('symlog')

    # plt.title('Tasks with Dense Rewards',fontsize=f_size)
    if normalised:
        plt.title('Tasks w/ Dense Rewards [Normalised]',fontsize=f_size)
        plt.ylabel('Normalised Return',fontsize=f_size)
        # plt.ylim(0,1.1)
    else: 
        # plt.title('Tasks w/ Dense Rewards [Unnormalised]',fontsize=f_size)
        plt.title('Tasks w/ Dense Rewards',fontsize=f_size)
        plt.ylabel('Return',fontsize=f_size)
        # plt.ylim(-240,15)
    # plt.ylabel('Return',fontsize=f_size)
    # plt.ylabel('Normalised Return',fontsize=f_size)
    plt.xlabel('Step [log-scale]',fontsize=f_size)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)

    # plt.xscale('logit')
    plt.xscale('log')

    plt.grid(True)
    plt.grid(which='minor',alpha=0.9) #, color='r'

    # save the figure
    if normalised:
        plt.savefig(
            'dense_rewards_normalised.png', 
            dpi=300, 
            bbox_inches='tight',
            format='png')
    else:
        plt.savefig(
            'dense_rewards_unnormalised.png', 
            dpi=300, 
            bbox_inches='tight',
            format='png')

    # plt.savefig(
    #     'dense_rewards.png', 
    #     dpi=300, 
    #     bbox_inches='tight',
    #     format='png')

    plt.show()
    xxx
    return 


###########################################
#### scripting #########
###########################################
leng = 50000
batch_a1 = pd.read_csv('1link100_d/1link100_0.csv')[:leng]
batch_a2 = pd.read_csv('1link100_d/1link100_1.csv')[:leng]
batch_a3 = pd.read_csv('1link100_d/1link100_2.csv')[:leng]
batch_a4 = pd.read_csv('1link100_d/1link100_3.csv')[:leng]
batch_a5 = pd.read_csv('1link100_d/1link100_4.csv')[:leng]

# batch_a1.at[0,'Step'] = 250
# batch_a2.at[0,'Step'] = 250
# batch_a3.at[0,'Step'] = 250
# batch_a4.at[0,'Step'] = 250
# batch_a5.at[0,'Step'] = 250

batch_b1 = pd.read_csv('1link170_d/1link170_0.csv')[:leng]
batch_b2 = pd.read_csv('1link170_d/1link170_1.csv')[:leng]
batch_b3 = pd.read_csv('1link170_d/1link170_2.csv')[:leng]
batch_b4 = pd.read_csv('1link170_d/1link170_3.csv')[:leng]
batch_b5 = pd.read_csv('1link170_d/1link170_4.csv')[:leng]

batch_c1 = pd.read_csv('2link_d/2link_0.csv')[3:leng]
batch_c2 = pd.read_csv('2link_d/2link_1.csv')[3:leng]
batch_c3 = pd.read_csv('2link_d/2link_2.csv')[3:leng]
batch_c4 = pd.read_csv('2link_d/2link_3.csv')[3:leng]
batch_c5 = pd.read_csv('2link_d/2link_4.csv')[3:leng]

# batch_c1.at[0,'Step'] = 250
# batch_c2.at[0,'Step'] = 250
# batch_c3.at[0,'Step'] = 250
# batch_c4.at[0,'Step'] = 250
# batch_c5.at[0,'Step'] = 250

# print('batch_1 : \n', batch_c2 )
# xxx

order_batch_a1 = batch_a1[['Step','Value']].sort_values(by='Step')
order_batch_a2 = batch_a2[['Step','Value']].sort_values(by='Step')
order_batch_a3 = batch_a3[['Step','Value']].sort_values(by='Step')
order_batch_a4 = batch_a4[['Step','Value']].sort_values(by='Step')
order_batch_a5 = batch_a5[['Step','Value']].sort_values(by='Step')

order_batch_b1 = batch_b1[['Step','Value']].sort_values(by='Step')
order_batch_b2 = batch_b2[['Step','Value']].sort_values(by='Step')
order_batch_b3 = batch_b3[['Step','Value']].sort_values(by='Step')
order_batch_b4 = batch_b4[['Step','Value']].sort_values(by='Step')
order_batch_b5 = batch_b5[['Step','Value']].sort_values(by='Step')

order_batch_c1 = batch_c1[['Step','Value']].sort_values(by='Step')
order_batch_c2 = batch_c2[['Step','Value']].sort_values(by='Step')
order_batch_c3 = batch_c3[['Step','Value']].sort_values(by='Step')
order_batch_c4 = batch_c4[['Step','Value']].sort_values(by='Step')
order_batch_c5 = batch_c5[['Step','Value']].sort_values(by='Step')

# print('order_batch_1: \n', order_batch_1)
# print('order_batch_2: \n', order_batch_2.shape)
# print('order_batch_3: \n', order_batch_3.shape)
# print('order_batch_4: \n', order_batch_4.shape)
# print('order_batch_5: \n', order_batch_5.shape)
# xxx

######################## Normalisation ################################
# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.RobustScaler(quantile_range=(5.0,95.0)) #,scale_to_unit=True

scaler_a1 = scaler.fit_transform(order_batch_a1['Value'].to_numpy().reshape(-1,1))
scaler_a2 = scaler.fit_transform(order_batch_a2['Value'].to_numpy().reshape(-1,1))
scaler_a3 = scaler.fit_transform(order_batch_a3['Value'].to_numpy().reshape(-1,1))
scaler_a4 = scaler.fit_transform(order_batch_a4['Value'].to_numpy().reshape(-1,1))
scaler_a5 = scaler.fit_transform(order_batch_a5['Value'].to_numpy().reshape(-1,1))
order_batch_a1['Scaled_Value'] = pd.DataFrame(scaler_a1)
order_batch_a2['Scaled_Value'] = pd.DataFrame(scaler_a2)
order_batch_a3['Scaled_Value'] = pd.DataFrame(scaler_a3)
order_batch_a4['Scaled_Value'] = pd.DataFrame(scaler_a4)
order_batch_a5['Scaled_Value'] = pd.DataFrame(scaler_a5)

scaler_b1 = scaler.fit_transform(order_batch_b1['Value'].to_numpy().reshape(-1,1))
scaler_b2 = scaler.fit_transform(order_batch_b2['Value'].to_numpy().reshape(-1,1))
scaler_b3 = scaler.fit_transform(order_batch_b3['Value'].to_numpy().reshape(-1,1))
scaler_b4 = scaler.fit_transform(order_batch_b4['Value'].to_numpy().reshape(-1,1))
scaler_b5 = scaler.fit_transform(order_batch_b5['Value'].to_numpy().reshape(-1,1))
order_batch_b1['Scaled_Value'] = pd.DataFrame(scaler_b1)
order_batch_b2['Scaled_Value'] = pd.DataFrame(scaler_b2)
order_batch_b3['Scaled_Value'] = pd.DataFrame(scaler_b3)
order_batch_b4['Scaled_Value'] = pd.DataFrame(scaler_b4)
order_batch_b5['Scaled_Value'] = pd.DataFrame(scaler_b5)

scaler_c1 = scaler.fit_transform(order_batch_c1['Value'].to_numpy().reshape(-1,1))
scaler_c2 = scaler.fit_transform(order_batch_c2['Value'].to_numpy().reshape(-1,1))
scaler_c3 = scaler.fit_transform(order_batch_c3['Value'].to_numpy().reshape(-1,1))
scaler_c4 = scaler.fit_transform(order_batch_c4['Value'].to_numpy().reshape(-1,1))
scaler_c5 = scaler.fit_transform(order_batch_c5['Value'].to_numpy().reshape(-1,1))
order_batch_c1['Scaled_Value'] = pd.DataFrame(scaler_c1)
order_batch_c2['Scaled_Value'] = pd.DataFrame(scaler_c2)
order_batch_c3['Scaled_Value'] = pd.DataFrame(scaler_c3)
order_batch_c4['Scaled_Value'] = pd.DataFrame(scaler_c4)
order_batch_c5['Scaled_Value'] = pd.DataFrame(scaler_c5)

# order_batch_a1['Scaled_Value'] = (order_batch_a1['Value']-order_batch_a1['Value'].min())/(order_batch_a1['Value'].max()-order_batch_a1['Value'].min())
# order_batch_a2['Scaled_Value'] = (order_batch_a2['Value']-order_batch_a2['Value'].min())/(order_batch_a2['Value'].max()-order_batch_a2['Value'].min())
# order_batch_a3['Scaled_Value'] = (order_batch_a3['Value']-order_batch_a3['Value'].min())/(order_batch_a3['Value'].max()-order_batch_a3['Value'].min())
# order_batch_a4['Scaled_Value'] = (order_batch_a4['Value']-order_batch_a4['Value'].min())/(order_batch_a4['Value'].max()-order_batch_a4['Value'].min())
# order_batch_a5['Scaled_Value'] = (order_batch_a5['Value']-order_batch_a5['Value'].min())/(order_batch_a5['Value'].max()-order_batch_a5['Value'].min())

# order_batch_b1['Scaled_Value'] = (order_batch_b1['Value']-order_batch_b1['Value'].min())/(order_batch_b1['Value'].max()-order_batch_b1['Value'].min())
# order_batch_b2['Scaled_Value'] = (order_batch_b2['Value']-order_batch_b2['Value'].min())/(order_batch_b2['Value'].max()-order_batch_b2['Value'].min())
# order_batch_b3['Scaled_Value'] = (order_batch_b3['Value']-order_batch_b3['Value'].min())/(order_batch_b3['Value'].max()-order_batch_b3['Value'].min())
# order_batch_b4['Scaled_Value'] = (order_batch_b4['Value']-order_batch_b4['Value'].min())/(order_batch_b4['Value'].max()-order_batch_b4['Value'].min())
# order_batch_b5['Scaled_Value'] = (order_batch_b5['Value']-order_batch_b5['Value'].min())/(order_batch_b5['Value'].max()-order_batch_b5['Value'].min())

# order_batch_c1['Scaled_Value'] = (order_batch_c1['Value']-order_batch_c1['Value'].min())/(order_batch_c1['Value'].max()-order_batch_c1['Value'].min())
# order_batch_c2['Scaled_Value'] = (order_batch_c2['Value']-order_batch_c2['Value'].min())/(order_batch_c2['Value'].max()-order_batch_c2['Value'].min())
# order_batch_c3['Scaled_Value'] = (order_batch_c3['Value']-order_batch_c3['Value'].min())/(order_batch_c3['Value'].max()-order_batch_c3['Value'].min())
# order_batch_c4['Scaled_Value'] = (order_batch_c4['Value']-order_batch_c4['Value'].min())/(order_batch_c4['Value'].max()-order_batch_c4['Value'].min())
# order_batch_c5['Scaled_Value'] = (order_batch_c5['Value']-order_batch_c5['Value'].min())/(order_batch_c5['Value'].max()-order_batch_c5['Value'].min())

# print(order_batch_c1['Value'].to_numpy())
# xxx

# scaler = preprocessing.MinMaxScaler()

# scaler_df = scaler.fit_transform(order_batch_c1['Value'])
# xxx
# scaler_df = pd.DataFrame(scaler_df) #, columns=[])

# print('order_batch_c1: ', order_batch_c1 )
# print('scaler_df: \n', scaler_df)
# xxx

###########################
# batch_list = [
#      order_batch_a1,
#      order_batch_a2,
#      order_batch_a3,
#      order_batch_a4,
#      order_batch_a5
#      ]
# #part2: ChatGPT
# start = max(
#      batch_list[0]['Step'].min(),
#      batch_list[1]['Step'].min(),
#      batch_list[2]['Step'].min(),
#      batch_list[3]['Step'].min(),
#      batch_list[4]['Step'].min(),
#      )

# end = min(
#      batch_list[0]['Step'].max(),
#      batch_list[1]['Step'].max(),
#      batch_list[2]['Step'].max(),
#      batch_list[3]['Step'].max(),
#      batch_list[4]['Step'].max(),
#      )

# grid = np.linspace(start,end,500)

# # print('start: ', start)
# # print('end: ', end)
# # print('grid: \n', grid)


# rewards = []
# for df in batch_list:
#      rewards.append(np.interp(grid,df['Step'],df['Value']))
#     #  print('rewards: ', rewards)
#     #  xx

# rewards = np.vstack(rewards)
# print('rewards: ', rewards)
# xxxx

# mean = rewards.mean(axis=0)
# std = rewards.std(axis=0)

# mean = roll_mean(mean)
# std = roll_mean(std)
# # print('mean: \n', mean)
# # print('std: \n', std)


# plotting(grid,mean,std)
# xxx

# plt.legend()
# plt.show()
############################


#merging datasets
batch_lists = {
     0 : [
            order_batch_a1,
            order_batch_a2,
            order_batch_a3,
            order_batch_a4,
            order_batch_a5
            ],
     1 : [
            order_batch_b1,
            order_batch_b2,
            order_batch_b3,
            order_batch_b4,
            order_batch_b5
            ],
     2 : [
            order_batch_c1,
            order_batch_c2,
            order_batch_c3,
            order_batch_c4,
            order_batch_c5
            ]
}

# print('batch_lists: \n', batch_lists)
# print('batch_lists[1]: \n', batch_lists[2])
# print(len(batch_lists))

grids = []
for i in range(len(batch_lists)):
    start = max(
     batch_lists[i][0]['Step'].min(),
     batch_lists[i][1]['Step'].min(),
     batch_lists[i][2]['Step'].min(),
     batch_lists[i][3]['Step'].min(),
     batch_lists[i][4]['Step'].min(),
     )

    end = min(
        batch_lists[i][0]['Step'].max(),
        batch_lists[i][1]['Step'].max(),
        batch_lists[i][2]['Step'].max(),
        batch_lists[i][3]['Step'].max(),
        batch_lists[i][4]['Step'].max(),
        )  
    # print('start: ', start)
    # print('end: ', end)
    grid = np.linspace(start,end,500)
    grids.append(grid)

means = []
stds = []
normalised = 0# True #
names = ['1link100','1link170','2link']
mean_per,max_per = [],[]
for i in range(len(batch_lists)):
    rewards = []
    for df in batch_lists[i]:
        # print('df: \n', df)
        # print('df: \n', df.describe())
        # print('mean_{}: '.format(names[i]), df.describe().loc['mean']['Value'])
        # print('max_{}: '.format(names[i]), df.describe().loc['max']['Value'])

        mean_per.append(df[-20:].describe().loc['mean']['Value'])
        max_per.append(df[-20:].describe().loc['max']['Value'])

        # mean_per.append(df.describe().loc['mean']['Value'])
        # max_per.append(df.describe().loc['max']['Value'])
         
        if normalised:
            rewards.append(np.interp(grids[i],df['Step'],df['Scaled_Value'])) #df['Value']
        else: 
            rewards.append(np.interp(grids[i],df['Step'],df['Value'])) 
        #  print('rewards: ', rewards)
        #  xx

    print('mean_{}: '.format(names[i]), np.asarray(mean_per).mean())
    print('max_{}: '.format(names[i]), np.asarray(max_per).mean())

    rewards = np.vstack(rewards)
    # print('rewards: ', rewards)

    #stats
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    #smoothened
    mean = roll_mean(mean)
    std = roll_mean(std)

    #collection
    means.append(mean)
    stds.append(std)

group_plot(grids,means,stds)