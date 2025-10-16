import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys #link_arm <>
sys.path.insert(0, "..") #link_arm <>

#modules
def roll_mean(array,window=100):
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
    
    j = 100
    plt.plot(grids[2][:-j],means[2][:-j],color="purple",label='2link_SAC')
    plt.fill_between(grids[2][:-j],
                    means[2][:-j] - stds[2][:-j],
                    means[2][:-j] + stds[2][:-j],
                    facecolor="purple", 
                    alpha=0.1,
                    )  
    
    plt.plot(grids[3][:-j],means[3][:-j],color="grey",label='2link_SAC+HER')
    plt.fill_between(grids[3][:-j],
                    means[3][:-j] - stds[3][:-j],
                    means[3][:-j] + stds[3][:-j],
                    facecolor="grey", 
                    alpha=0.1,
                    )  

    
    plt.legend(fontsize=f_size) #plt.legend(loc='lower right',fontsize=f_size)
    
    # plt.yscale('symlog')
    # plt.xscale('symlog')

    plt.title('Tasks with Sparse Rewards',fontsize=f_size)
    plt.ylabel('Return',fontsize=f_size)
    plt.xlabel('Step',fontsize=f_size)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)

    plt.ticklabel_format(axis='x',style='sci',scilimits=(3,3))


    plt.grid(True)
    plt.grid(which='minor', color='r') #,alpha=0.9

    plt.show()
    xxx
    # save the figure
    
    # plt.savefig(
    #     'sparse_rewards.png', 
    #     dpi=300, 
    #     bbox_inches='tight',
    #     format='png')

    plt.show()
    xxx
    return 


###########################################
#### scripting #########
###########################################
leng = 10000 #10000
batch_a1 = pd.read_csv('1link100_s/1link100_0.csv')[:leng]
batch_a2 = pd.read_csv('1link100_s/1link100_1.csv')[:leng]
batch_a3 = pd.read_csv('1link100_s/1link100_2.csv')[:leng]
batch_a4 = pd.read_csv('1link100_s/1link100_3.csv')[:leng]
batch_a5 = pd.read_csv('1link100_s/1link100_4.csv')[:leng]

batch_b1 = pd.read_csv('1link170_s/1link170_0.csv')[:leng]
batch_b2 = pd.read_csv('1link170_s/1link170_1.csv')[:leng]
batch_b3 = pd.read_csv('1link170_s/1link170_2.csv')[:leng]
batch_b4 = pd.read_csv('1link170_s/1link170_3.csv')[:leng]
batch_b5 = pd.read_csv('1link170_s/1link170_4.csv')[:leng]

# batch_c1 = pd.read_csv('2link_s/2link_SAC0.csv')[:leng]
batch_c1 = pd.read_csv('2link_s/2link_SAC_1.csv')[:leng]
batch_c2 = pd.read_csv('2link_s/2link_SAC_2.csv')[:leng]
batch_c3 = pd.read_csv('2link_s/2link_SAC_3.csv')[:leng]
batch_c4 = pd.read_csv('2link_s/2link_SAC_4.csv')[:leng]
batch_c5 = pd.read_csv('2link_s/2link_SAC_5.csv')[:leng]

# batch_d1 = pd.read_csv('2link_s/2link_SAC+HER0.csv')[:leng]
batch_d1 = pd.read_csv('2link_s/2link_SAC+HER_1.csv')[:leng]
batch_d2 = pd.read_csv('2link_s/2link_SAC+HER_2.csv')[:leng]
batch_d3 = pd.read_csv('2link_s/2link_SAC+HER_3.csv')[:leng]
batch_d4 = pd.read_csv('2link_s/2link_SAC+HER_4.csv')[:leng]
batch_d5 = pd.read_csv('2link_s/2link_SAC+HER_5.csv')[:leng]


# print('batch_1 : \n', batch_1 )
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

order_batch_d1 = batch_d1[['Step','Value']].sort_values(by='Step')
order_batch_d2 = batch_d2[['Step','Value']].sort_values(by='Step')
order_batch_d3 = batch_d3[['Step','Value']].sort_values(by='Step')
order_batch_d4 = batch_d4[['Step','Value']].sort_values(by='Step')
order_batch_d5 = batch_d5[['Step','Value']].sort_values(by='Step')

# print('order_batch_1: \n', order_batch_1)
# print('order_batch_2: \n', order_batch_2.shape)
# print('order_batch_3: \n', order_batch_3.shape)
# print('order_batch_4: \n', order_batch_4.shape)
# print('order_batch_5: \n', order_batch_5.shape)
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
     2 : [ #NATIVE
            order_batch_c1,
            order_batch_c2,
            order_batch_c3,
            order_batch_c4,
            order_batch_c5
            ],
     3 : [ #HER
            order_batch_d1,
            order_batch_d2,
            order_batch_d3,
            order_batch_d4,
            order_batch_d5
            ]
}

# Completion of Data collection
all_data_is_there = True #False

# print('batch_lists: \n', batch_lists)
# print('batch_lists[1]: \n', batch_lists[2])
# print(len(batch_lists))

grids = []
for i in range(len(batch_lists)):

    if all_data_is_there == True:
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
    else:
        # print('i: ', i) 
        if i == 2 or i == 3: 
            # print('two')
            print('two-3')
            start = batch_lists[i][0]['Step'].min()
            end = batch_lists[i][0]['Step'].max()
            # start = max(
            #     batch_lists[i][0]['Step'].min(),
            #     # batch_lists[i][1]['Step'].min(),
            #     # batch_lists[i][2]['Step'].min(),
            #     # batch_lists[i][3]['Step'].min(),
            #     # batch_lists[i][4]['Step'].min(),
            #     )

            # end = min(
            #     batch_lists[i][0]['Step'].max(),
            #     # batch_lists[i][1]['Step'].max(),
            #     # batch_lists[i][2]['Step'].max(),
            #     # batch_lists[i][3]['Step'].max(),
            #     # batch_lists[i][4]['Step'].max(),
            #     ) 

        else: 
            print('i: ', i)
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
    grid = np.linspace(start,end,2000)
    grids.append(grid)
    # xxx

means = []
stds = []
names = ['1link100','1link170','2link','2linkHER']
# mean_per,max_per = [],[]
for i in range(len(batch_lists)):
    mean_per,max_per = [],[]
    # print(i)
    # i = 2
    rewards = []
    for df in batch_lists[i]:
        # print('i: ', i)
        rewards.append(np.interp(grids[i],df['Step'],df['Value']))
        #  print('rewards: ', rewards)

        # if i == 2:
        #     print(df[-20:])
        #     print(df[-20:].describe().loc['mean']['Value'])
        
        mean_per.append(df[-20:].describe().loc['mean']['Value'])
        max_per.append(df[-20:].describe().loc['max']['Value'])
    # print(df[-20:])

    # xxx
    # print(mean_per)
    # xxx
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
    # xxx

# xxx
group_plot(grids,means,stds)